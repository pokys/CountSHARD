"""Effect of suppressing nested GeCo2 boxes (a box mostly inside a stronger one) on the counts,
using the objectness/ltrb maps saved by run_onnx.py (maps-<size>-<variant>-<case>.npz).

Usage: python -I containment_eval.py <cases-dir> <maps-dir> <size> <variant> [ratio ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_onnx import maxpool3, nms  # noqa: E402
from ref_pytorch import load_image, preprocess  # noqa: E402


def detect(obj, ltrb, wh, size, contain=None):
    o = obj[0, 0]
    H, W = o.shape
    peaks = (maxpool3(o) == o) & (o > o.max() / 8)
    ys, xs = np.nonzero(peaks)
    v = o[ys, xs]
    off = ltrb[0][:, ys, xs].T
    c = np.stack([xs / W, ys / H], 1).astype(np.float32)
    b = np.concatenate([c - off[:, :2], c + off[:, 2:]], 1)
    k = nms(b, v, 0.5)
    b, v = np.clip(b[k], 0, 1), v[k]
    if contain is not None:
        keep = []
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        for i in np.argsort(-v, kind='stable'):
            ok = True
            for j in keep:
                iw = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
                ih = min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1])
                if iw > 0 and ih > 0 and iw * ih / max(1e-9, min(area[i], area[j])) > contain:
                    ok = False
                    break
            if ok:
                keep.append(i)
        b = b[keep]
    cc = (b[:, :2] + b[:, 2:]) / 2 * size
    return int(((cc[:, 0] < wh[0]) & (cc[:, 1] < wh[1])).sum())


cases_dir, maps_dir, size, variant = sys.argv[1:5]
size = int(size)
ratios = [float(r) for r in sys.argv[5:]] or [0.8, 0.7, 0.6]
cases = json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8'))
groups = {'synthetic': [], 'fsc': [], 'mosaic': []}
rows = []
for c in cases:
    f = os.path.join(maps_dir, f"maps-{size}-{variant}-{c['name'].replace(' ', '_').replace('×', 'x')}.npz")
    if not os.path.exists(f):
        continue
    m = np.load(f)
    _, _, _, wh = preprocess(load_image(c), c['exemplars'], size, 80.0 * size / 1024)
    counts = [detect(m['obj'], m['ltrb'], wh, size)] + [detect(m['obj'], m['ltrb'], wh, size, r) for r in ratios]
    g = 'synthetic' if 'truth' in c else 'mosaic' if c.get('mosaic') else 'fsc'
    groups[g].append([x - c['gt'] for x in counts])
    rows.append((c['name'], c['gt'], counts))
print('| case | GT | NMS only | ' + ' | '.join(f'+ nested > {r}' for r in ratios) + ' |')
for n, gt, counts in rows:
    print(f'| {n} | {gt} | ' + ' | '.join(str(x) for x in counts) + ' |')
for g, errs in groups.items():
    if errs:
        e = np.abs(np.array(errs)).mean(0)
        print(f'MAE {g} ({len(errs)}): ' + ', '.join(f'{x:.1f}' for x in e))
