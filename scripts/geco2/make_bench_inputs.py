"""Test inputs for the phone bench: for each case and size, the photo already scaled for the
model (PNG, content only; the page normalises and pads), the exemplar boxes in model pixels and
the count onnxruntime CPU gets from exactly that PNG (so the phone result can be compared 1:1).

Usage: python -I make_bench_inputs.py <onnx-dir> <cases-dir> <out-dir> <size,size> <name-prefix>...
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pytorch import load_image  # noqa: E402
from run_onnx import detect, run_case, sessions  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

d, cases_dir, out = sys.argv[1:4]
sizes = [int(s) for s in sys.argv[4].split(',')]
prefixes = sys.argv[5:]
os.makedirs(out, exist_ok=True)
cases = json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8'))
index = []
for S in sizes:
    sess = sessions(d, S, 'w8', 4)
    for c in cases:
        if not any(c['name'].startswith(p) for p in prefixes):
            continue
        img = load_image(c)
        # Same scale rule as ref_pytorch.preprocess: longer side -> S, exemplars <= 80*S/1024 px.
        H, W = img.shape[:2]
        s = S / max(H, W)
        ex = np.array(c['exemplars'], np.float32) * s
        a = ((ex[:, 2] - ex[:, 0]).mean() + (ex[:, 3] - ex[:, 1]).mean()) / 2
        s = min(1.0, (80.0 * S / 1024) / a) * s
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float()[None]
        r = F.interpolate(t, scale_factor=s, mode='bilinear', align_corners=False)[0]
        rgb = r.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        h, w = rgb.shape[:2]
        key = c['name'].split(' ')[0] + f'-{S}'
        Image.fromarray(rgb).save(os.path.join(out, key + '.png'), optimize=True)
        x = np.zeros((1, 3, S, S), np.float32)
        x[0, :, :h, :w] = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        boxes = (np.array(c['exemplars'], np.float32) * s)[None]
        obj, ltrb, _, _ = run_case(sess, x, boxes)
        n = len(detect(obj, ltrb, (w, h), S)[0])
        meta = {'key': key, 'name': c['name'], 'size': S, 'gt': c['gt'], 'content': [w, h],
                'boxes': boxes.reshape(-1).round(2).tolist(), 'cpu_count': n}
        json.dump(meta, open(os.path.join(out, key + '.json'), 'w', encoding='utf8'), ensure_ascii=False)
        index.append({k: meta[k] for k in ('key', 'name', 'size', 'gt', 'cpu_count')})
        print(key, 'gt', c['gt'], 'cpu', n, flush=True)
json.dump(index, open(os.path.join(out, 'index.json'), 'w', encoding='utf8'), ensure_ascii=False)
