"""Quantise the exported LOCA graphs and compare every variant with the PyTorch reference.

usage: python -I quant_verify.py <onnx_dir>
"""
import sys, os, time
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
import torch

d = sys.argv[1]


def weight_only_int8(src, dst, min_elems=1024):
    """Store large Conv/MatMul/Gemm weights as int8 (symmetric, per output channel) followed by
    DequantizeLinear. Compute stays fp32, so every execution provider can run it."""
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    consumers = {}
    for n in g.node:
        for k, inp in enumerate(n.input):
            consumers.setdefault(inp, []).append((n, k))
    new_inits, new_nodes, done = [], [], 0
    for name, init in list(inits.items()):
        uses = consumers.get(name, [])
        if not uses:
            continue
        axis = None
        ok = True
        for n, k in uses:
            if n.op_type == 'Conv' and k == 1:
                a = 0
            elif n.op_type == 'MatMul' and k == 1:
                a = 1
            elif n.op_type == 'Gemm' and k == 1:
                tb = next((at.i for at in n.attribute if at.name == 'transB'), 0)
                a = 0 if tb else 1
            else:
                ok = False; break
            if axis is not None and axis != a:
                ok = False; break
            axis = a
        w = numpy_helper.to_array(init)
        if not ok or w.dtype != np.float32 or w.size < min_elems or w.ndim < 2:
            continue
        red = tuple(i for i in range(w.ndim) if i != axis)
        amax = np.abs(w).max(axis=red)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        shape = [1] * w.ndim; shape[axis] = -1
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


variants = {'fp32': ('loca_features_fp32.onnx', 'loca_head_fp32.onnx')}
for part in ('features', 'head'):
    n = weight_only_int8(os.path.join(d, f'loca_{part}_fp32.onnx'), os.path.join(d, f'loca_{part}_w8.onnx'))
    print(part, 'weight-only int8 tensors:', n)
variants['w8 (weight-only int8, fp32 compute)'] = ('loca_features_w8.onnx', 'loca_head_w8.onnx')
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    for part in ('features', 'head'):
        quantize_dynamic(os.path.join(d, f'loca_{part}_fp32.onnx'), os.path.join(d, f'loca_{part}_dyn8.onnx'), weight_type=QuantType.QUInt8)
    variants['dyn8 (dynamic int8, ConvInteger/MatMulInteger)'] = ('loca_features_dyn8.onnx', 'loca_head_dyn8.onnx')
except Exception as e:
    print('dynamic quantisation failed:', e)

ref = dict(np.load(os.path.join(d, 'reference.npz')))
samples = torch.load(os.path.join(d, 'samples.pt'), weights_only=False)
so = ort.SessionOptions(); so.intra_op_num_threads = 1

for label, (fp, hp) in variants.items():
    size = (os.path.getsize(os.path.join(d, fp)) + os.path.getsize(os.path.join(d, hp))) / 1e6
    try:
        fs = ort.InferenceSession(os.path.join(d, fp), so, providers=['CPUExecutionProvider'])
        hs = ort.InferenceSession(os.path.join(d, hp), so, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(label, 'load failed', e); continue
    rows = []
    for n, x, b, gt in samples:
        t = time.time()
        f = fs.run(None, {'image': x.numpy()})[0]
        t1 = time.time()
        den = hs.run(None, {'features': f, 'exemplar_features': f, 'bboxes': b.numpy()})[0]
        t2 = time.time()
        r = ref[n]
        rows.append((n, gt, float(r.sum()), float(den.sum()), float(np.abs(den - r).max()), float(np.abs(den - r).max() / max(r.max(), 1e-9)), (t1 - t) * 1000, (t2 - t1) * 1000))
    print(f'\n## {label}: {size:.1f} MB')
    print('| image | GT | torch count | onnx count | count diff | max abs density diff | rel. to max | features ms | head ms |')
    for n, gt, rc, oc, mad, rel, tf, th in rows:
        print(f'| {n} | {gt} | {rc:.2f} | {oc:.2f} | {oc-rc:+.3f} | {mad:.2e} | {rel:.2%} | {tf:.0f} | {th:.0f} |')
