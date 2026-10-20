"""GeCo2 PyTorch reference on the dumped evaluation cases (scripts/geco2/dump-cases.ts).

Preprocessing and counting follow GeCo2's own inference.py / demo: ImageNet-normalise, scale
the longer side to 1024 and further down so the mean exemplar side is <= 80 px, pad to
1024 x 1024 (bottom/right, zeros); peaks of the 3x3-max-pooled objectness above max/8;
keep score > 0.11 * max; NMS 0.5 on the (SAM-refined) boxes; drop centres in the padding.

Also records the count with the *unrefined* regression boxes (no SAM mask decoder), which is
what a browser build without the mask decoder graph would produce.

Usage: python -I ref_pytorch.py <geco2-code> <ckpt> <cases-dir> <out.json> [name-filter]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geco2_env  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(case):
    img = np.asarray(Image.open(case['file']).convert('RGB'))
    if case.get('mosaic'):
        img = np.concatenate([np.concatenate([img, img], 1)] * 2, 0)
    return img


def preprocess(img, exemplars, size=1024, max_ex=80.0):
    """Returns (padded [1,3,S,S], boxes [1,K,4] in model px, scale, (w, h) of the content)."""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = (t - MEAN) / STD
    _, H, W = t.shape
    s = size / max(H, W)
    b = torch.tensor(exemplars, dtype=torch.float32) * s
    a_dim = ((b[:, 2] - b[:, 0]).mean() + (b[:, 3] - b[:, 1]).mean()) / 2
    s = min(1.0, max_ex / a_dim.item()) * s
    r = F.interpolate(t[None], scale_factor=s, mode='bilinear', align_corners=False)
    h, w = r.shape[2:]
    p = F.pad(r, (0, size - w, 0, size - h))
    return p, torch.tensor(exemplars, dtype=torch.float32)[None] * s, s, (w, h)


def select(box_v, boxes, content_wh, size, ops, thr=0.11, iou=0.5):
    """GeCo2 eval post-processing. boxes normalised [N,4] (x1,y1,x2,y2 / size)."""
    if box_v.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros(0)
    m = box_v > box_v.max() * thr
    keep = ops.nms(boxes[m], box_v[m], iou)
    b = boxes[m][keep].clamp(0, 1)
    v = box_v[m][keep]
    c = (b[:, :2] + b[:, 2:]) / 2 * size
    ok = (c[:, 0] < content_wh[0]) & (c[:, 1] < content_wh[1])
    return b[ok], v[ok]


def main():
    code, ckpt, cases_dir, out = sys.argv[1:5]
    filt = sys.argv[5] if len(sys.argv) > 5 else ''
    env = geco2_env.load(code, ckpt)
    model = env.model
    raw = {}
    orig_mask = model.sam_mask.forward

    def spy(feats, outputs):
        raw['boxes'] = outputs[0]['pred_boxes'][0].clone()
        raw['box_v'] = outputs[0]['box_v'][0].clone()
        return orig_mask(feats, outputs)

    model.sam_mask.forward = spy
    cases = json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8'))
    res = []
    for c in cases:
        if filt and filt not in c['name']:
            continue
        img = load_image(c)
        x, bx, s, wh = preprocess(img, c['exemplars'])
        t0 = time.time()
        with torch.no_grad():
            outputs, _, centerness, _, _ = model(x, bx)
        dt = time.time() - t0
        o = outputs[0]
        refined = o['pred_boxes'][0] if o['pred_boxes'].dim() == 3 else o['pred_boxes']
        refined = refined.reshape(-1, 4)
        # GeCo2 eval: boxes from the mask decoder (pixels / 1024), scores = objectness box_v.
        b1, v1 = select(raw['box_v'], refined, wh, 1024, env.ops)
        b0, v0 = select(raw['box_v'], raw['boxes'], wh, 1024, env.ops)
        r = {
            'name': c['name'], 'gt': c['gt'], 'scale': s, 'ms': round(dt * 1000),
            'candidates': int(raw['box_v'].numel()),
            'count': len(b1), 'count_raw': len(b0),
            'boxes': (b1 * 1024 / s).round(decimals=1).tolist(),
            'boxes_raw': (b0 * 1024 / s).round(decimals=1).tolist(),
            'scores_raw': v0.tolist(),
            'objectness_max': float(centerness.max()),
        }
        res.append(r)
        print(f"{c['name']:28s} gt {c['gt']:5d}  geco2 {r['count']:5d}  raw-boxes {r['count_raw']:5d}  scale {s:.2f}  {dt:.1f}s", flush=True)
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    main()
