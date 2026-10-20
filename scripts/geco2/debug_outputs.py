"""Bisect execution-provider differences: add intermediate tensors as graph outputs.

  python -I debug_outputs.py list <model.onnx> [pattern]        # node names / op types
  python -I debug_outputs.py add <model.onnx> <out.onnx> <regex> [every]
      every node whose name matches <regex> (every n-th match) gets its first output exposed;
      writes <out.onnx> and <out.onnx>.names.json (output names in order).
"""
import json
import re
import sys

import onnx
from onnx import shape_inference

cmd = sys.argv[1]
if cmd == 'ref':
    # python -I debug_outputs.py ref <debug.onnx> <inputs-dir> <key> <graph: backbone|head>
    # CPU fingerprints of every exposed output: shape, sum, abs-sum, every 997th value.
    import os
    import numpy as np
    import onnxruntime as ort
    dbg, d, key, graph = sys.argv[2:6]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(dbg, so, providers=['CPUExecutionProvider'])
    meta = json.load(open(os.path.join(d, key + '.json')))
    S = meta['size']
    rd = lambda n, shape: np.fromfile(os.path.join(d, f'{key}.{n}.bin'), np.float32).reshape(shape)
    if graph == 'backbone':
        feeds = {'image': rd('img', (1, 3, S, S))}
    else:
        feeds = {'feat': rd('feat', (1, 256, S // 16, S // 16)), 'fpn1': rd('fpn1', (1, 256, S // 8, S // 8)),
                 'fpn0': rd('fpn0', (1, 256, S // 4, S // 4))}
        for k in ('proto0', 'proto1', 'proto2'):
            feeds[k] = rd(k, (1, -1, 256))
    names = [o.name for o in s.get_outputs()]
    res = {}
    for n, v in zip(names, s.run(None, feeds)):
        f = v.reshape(-1).astype(np.float64)
        res[n] = {'shape': list(v.shape), 'sum': float(f.sum()), 'abs': float(np.abs(f).sum()), 'sample': f[::997].tolist()}
    json.dump(res, open(os.path.join(d, f'{key}.dbg-{graph}.json'), 'w'))
    print(len(res), 'fingerprints')
    sys.exit()
m = onnx.load(sys.argv[2])
if cmd == 'list':
    pat = sys.argv[3] if len(sys.argv) > 3 else ''
    for n in m.graph.node:
        if pat in n.name or pat == n.op_type:
            print(n.op_type, n.name, list(n.output))
    sys.exit()
out, rx = sys.argv[3], re.compile(sys.argv[4])
every = int(sys.argv[5]) if len(sys.argv) > 5 else 1
m = shape_inference.infer_shapes(m)
types = {v.name: v for v in list(m.graph.value_info)}
names, k = [], 0
existing = {o.name for o in m.graph.output}
for n in m.graph.node:
    if n.op_type in ('Constant', 'Shape') or not rx.search(n.name):
        continue
    k += 1
    if k % every:
        continue
    o = n.output[0]
    if o in existing:
        continue
    vi = types.get(o)
    if vi is None or vi.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        continue
    m.graph.output.append(vi)
    names.append(o)
onnx.save(m, out)
json.dump(names, open(out + '.names.json', 'w'))
print(len(names), 'outputs added')
