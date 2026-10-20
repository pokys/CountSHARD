"""Draw GeCo2 detections (orange) and exemplars (red) for a results JSON from ref_pytorch.py /
run_onnx.py. Usage: python -I draw.py <cases-dir> <results.json> <out-dir> [name-filter] [key]"""
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_pytorch import load_image  # noqa: E402

cases_dir, res_path, out = sys.argv[1:4]
filt = sys.argv[4] if len(sys.argv) > 4 else ''
key = sys.argv[5] if len(sys.argv) > 5 else 'boxes'
os.makedirs(out, exist_ok=True)
cases = {c['name']: c for c in json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8'))}
for r in json.load(open(res_path, encoding='utf8')):
    if filt and filt not in r['name']:
        continue
    c = cases[r['name']]
    im = Image.fromarray(load_image(c)).convert('RGB')
    d = ImageDraw.Draw(im)
    for b in r[key]:
        d.rectangle(b, outline=(255, 150, 0), width=1)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 150, 0))
    for b in c['exemplars']:
        d.rectangle(b, outline=(255, 0, 0), width=2)
    d.text((5, 5), f"{r['name']}: {len(r[key])} (GT {r['gt']})", fill=(255, 255, 0))
    im.save(os.path.join(out, r['name'].replace(' ', '_').replace('×', 'x').replace('(', '').replace(')', '') + '.png'))
