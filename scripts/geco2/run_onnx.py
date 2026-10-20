"""Quantise the exported GeCo2 graphs (weight-only int8) and run them with onnxruntime on the
dumped cases; detection post-processing in NumPy (what the app does in TypeScript).

  python -I run_onnx.py quant <onnx-dir> <size>
  python -I run_onnx.py run <onnx-dir> <size> <fp32|w8> <cases-dir> <out.json> [threads] [name-filter]

Post-processing = GeCo2's eval: peaks of the 3x3-max-pooled objectness above max/8, boxes =
centre (col/W, row/H) -/+ ltrb, keep score > 0.11 * max, greedy NMS (IoU 0.5), clamp to
[0, 1], drop centres in the padding. Env NMS_IOU / SCORE_THR change the last two steps.
"""
import json
import os
import sys
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def weight_only_int8(src, dst, min_elems=1024):
    """Conv/MatMul/Gemm weights -> int8 (symmetric, per output channel) + DequantizeLinear.
    Same scheme as scripts/loca/quant_verify.py: compute stays fp32."""
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    # Weights used in several places (banded blocks) are exported behind Identity nodes:
    # point the consumers at the initializer itself and drop the Identity.
    alias = {n.output[0]: n.input[0] for n in g.node if n.op_type == 'Identity' and n.input[0] in inits}
    if alias:
        graph_outputs = {o.name for o in g.output}
        keep = [n for n in g.node if not (n.op_type == 'Identity' and n.output[0] in alias and n.output[0] not in graph_outputs)]
        for n in keep:
            for k, inp in enumerate(n.input):
                if inp in alias and inp not in graph_outputs:
                    n.input[k] = alias[inp]
        del g.node[:]
        g.node.extend(keep)
    consumers = {}
    for n in g.node:
        for k, inp in enumerate(n.input):
            consumers.setdefault(inp, []).append((n, k))
    new_inits, new_nodes, done = [], [], 0
    for name, init in list(inits.items()):
        uses = consumers.get(name, [])
        if not uses:
            continue
        axis, ok = None, True
        for n, k in uses:
            if n.op_type == 'Conv' and k == 1:
                a = 0
            elif n.op_type == 'MatMul' and k == 1:
                a = 1
            elif n.op_type == 'Gemm' and k == 1:
                a = 0 if next((at.i for at in n.attribute if at.name == 'transB'), 0) else 1
            else:
                ok = False
                break
            if axis is not None and axis != a:
                ok = False
                break
            axis = a
        w = numpy_helper.to_array(init)
        if not ok or w.dtype != np.float32 or w.size < min_elems or w.ndim < 2:
            continue
        red = tuple(i for i in range(w.ndim) if i != axis)
        amax = np.abs(w).max(axis=red)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        g.initializer.remove(init)
        new_inits += [numpy_helper.from_array(q, name + '_q'), numpy_helper.from_array(scale, name + '_s'),
                      numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + '_z')]
        new_nodes.append(helper.make_node('DequantizeLinear', [name + '_q', name + '_s', name + '_z'], [name], axis=axis, name=name + '_dq'))
        done += 1
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    onnx.save(m, dst)
    return done


def maxpool3(o):
    p = np.pad(o, 1, constant_values=-np.inf)
    h, w = o.shape
    return np.max(np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)]), 0)


def nms(boxes, scores, thr):
    order = np.argsort(-scores, kind='stable')
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= thr]
    return np.array(keep, dtype=np.int64)


def detect(obj, ltrb, content_wh, size, thr=None, iou=None):
    thr = float(os.environ.get('SCORE_THR', '0.11')) if thr is None else thr
    iou = float(os.environ.get('NMS_IOU', '0.5')) if iou is None else iou
    o = obj[0, 0]
    H, W = o.shape
    peaks = (maxpool3(o) == o) & (o > o.max() / 8)
    ys, xs = np.nonzero(peaks)
    v = o[ys, xs]
    if v.size == 0:
        return np.zeros((0, 4), np.float32), v
    off = ltrb[0][:, ys, xs].T
    c = np.stack([xs / W, ys / H], 1).astype(np.float32)
    boxes = np.concatenate([c - off[:, :2], c + off[:, 2:]], 1)
    m = v > v.max() * thr
    boxes, v = boxes[m], v[m]
    k = nms(boxes, v, iou)
    boxes, v = np.clip(boxes[k], 0, 1), v[k]
    cc = (boxes[:, :2] + boxes[:, 2:]) / 2 * size
    ok = (cc[:, 0] < content_wh[0]) & (cc[:, 1] < content_wh[1])
    return boxes[ok], v[ok]


def sessions(d, size, variant, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return [ort.InferenceSession(os.path.join(d, f'geco2-{k}-{size}-{variant}.onnx'), so, providers=['CPUExecutionProvider'])
            for k in ('backbone', 'exemplars', 'head')]


def run_case(sess, x, bx):
    b, e, h = sess
    t = time.time()
    feat, fpn1, fpn0 = b.run(None, {'image': x})
    t1 = time.time()
    p0, p1, p2 = e.run(None, {'feat': feat, 'fpn1': fpn1, 'fpn0': fpn0, 'boxes': bx})
    obj, ltrb = h.run(None, {'feat': feat, 'fpn1': fpn1, 'fpn0': fpn0, 'proto0': p0, 'proto1': p1, 'proto2': p2})
    t2 = time.time()
    return obj, ltrb, (t1 - t) * 1000, (t2 - t1) * 1000


def main():
    cmd = sys.argv[1]
    if cmd == 'quant':
        d, size = sys.argv[2], sys.argv[3]
        for k in ('backbone', 'exemplars', 'head'):
            src = os.path.join(d, f'geco2-{k}-{size}-fp32.onnx')
            dst = os.path.join(d, f'geco2-{k}-{size}-w8.onnx')
            n = weight_only_int8(src, dst)
            print(f'{k}: {n} int8 tensors, {os.path.getsize(src) / 1e6:.1f} -> {os.path.getsize(dst) / 1e6:.1f} MB')
        return
    if cmd == 'dump':
        # Inputs + CPU outputs for the browser bench (scripts/geco2/bench.html):
        # <name>-<size>.img.bin (float32 image), .obj.bin (CPU objectness), .json (boxes, count).
        d, size, variant, cases_dir, out_dir, filt = sys.argv[2:8]
        size = int(size)
        from ref_pytorch import load_image, preprocess
        sess = sessions(d, size, variant, 4)
        os.makedirs(out_dir, exist_ok=True)
        for c in json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8')):
            if filt not in c['name']:
                continue
            x, bx, s, wh = preprocess(load_image(c), c['exemplars'], size, 80.0 * size / 1024)
            obj, ltrb, _, _ = run_case(sess, x.numpy(), bx.numpy())
            boxes, _ = detect(obj, ltrb, wh, size)
            key = c['name'].split(' ')[0] + f'-{size}'
            fb = sess[0].run(None, {'image': x.numpy()})
            for k, v in zip(('feat', 'fpn1', 'fpn0'), fb):
                v.astype(np.float32).tofile(os.path.join(out_dir, f'{key}.{k}.bin'))
            pr = sess[1].run(None, {'feat': fb[0], 'fpn1': fb[1], 'fpn0': fb[2], 'boxes': bx.numpy()})
            for k, v in zip(('proto0', 'proto1', 'proto2'), pr):
                v.astype(np.float32).tofile(os.path.join(out_dir, f'{key}.{k}.bin'))
            x.numpy().astype(np.float32).tofile(os.path.join(out_dir, key + '.img.bin'))
            obj.astype(np.float32).tofile(os.path.join(out_dir, key + '.obj.bin'))
            json.dump({'name': c['name'], 'gt': c['gt'], 'size': size, 'boxes': bx.numpy().reshape(-1).tolist(),
                       'content': list(wh), 'cpu_count': int(len(boxes))}, open(os.path.join(out_dir, key + '.json'), 'w'))
            print(key, 'cpu count', len(boxes))
        return
    d, size, variant, cases_dir, out = sys.argv[2:7]
    size = int(size)
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    filt = sys.argv[8] if len(sys.argv) > 8 else ''
    from ref_pytorch import load_image, preprocess
    sess = sessions(d, size, variant, threads)
    res = []
    for c in json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8')):
        if filt and filt not in c['name']:
            continue
        x, bx, s, wh = preprocess(load_image(c), c['exemplars'], size, 80.0 * size / 1024)
        obj, ltrb, tb, th = run_case(sess, x.numpy(), bx.numpy())
        boxes, v = detect(obj, ltrb, wh, size)
        r = {'name': c['name'], 'gt': c['gt'], 'scale': s, 'count': int(len(boxes)),
             'boxes': np.round(boxes * size / s, 1).tolist(), 'scores': v.tolist(),
             'ms_backbone': round(tb), 'ms_head': round(th)}
        res.append(r)
        print(f"{c['name']:28s} gt {c['gt']:5d}  {variant} {r['count']:5d}  backbone {tb / 1000:.1f}s head {th / 1000:.1f}s", flush=True)
        np.savez_compressed(os.path.join(os.path.dirname(out), f"maps-{size}-{variant}-{c['name'].replace(' ', '_').replace('×', 'x')}.npz"), obj=obj, ltrb=ltrb)
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    main()
