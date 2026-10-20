"""Markdown tables: GeCo2 (results JSON from ref_pytorch.py / run_onnx.py) vs LOCA (the Mode B
tables in docs/testing.md, same cases). Synthetic scenes also get per-object precision/recall:
a detection is correct if its box centre lies within 0.5 r of an unmatched true centre.

Usage: python -I compare.py <cases-dir> <testing.md> <label>=<results.json> [...]
"""
import json
import math
import os
import re
import sys

cases_dir, testing = sys.argv[1:3]
runs = [a.split('=', 1) for a in sys.argv[3:]]
cases = {c['name']: c for c in json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8'))}
res = {label: {r['name']: r for r in json.load(open(p, encoding='utf8'))} for label, p in runs}

# LOCA rows: "| name | GT | size | ex | squash sum | >5% | tiles | scale | tiles: sum | ms |"
loca = {}
for line in open(testing, encoding='utf8'):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    if len(cells) == 10 and re.match(r'^\d+$', cells[1]):
        loca[re.sub(r'\s+', '', cells[0])] = (float(cells[4]), float(cells[8]))


def match(truth, boxes):
    dets = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    pairs = sorted((math.hypot(d[0] - t['x'], d[1] - t['y']), i, j) for i, d in enumerate(dets)
                   for j, t in enumerate(truth) if abs(d[0] - t['x']) < t['r'] and abs(d[1] - t['y']) < t['r'])
    ud, ut, tp = set(), set(), 0
    for dist, i, j in pairs:
        if i in ud or j in ut or dist > 0.5 * truth[j]['r']:
            continue
        ud.add(i)
        ut.add(j)
        tp += 1
    return tp, len(dets) - tp, len(truth) - tp


def table(names, title, synthetic):
    labels = [l for l, _ in runs]
    head = f'| {title} | GT | LOCA squash | LOCA tiles (app) | ' + ' | '.join(f'GeCo2 {l}' for l in labels)
    if synthetic:
        head += ' | ' + ' | '.join(f'{l}: TP / FP / FN' for l in labels)
    print(head + ' |')
    print('|' + '---|' * head.count('|'))
    err = {k: [] for k in ['sq', 'ti'] + labels}
    for n in names:
        c = cases[n]
        key = re.sub(r'\s+', '', n)
        lq, lt = loca.get(key, (float('nan'), float('nan')))
        row = f'| {n} | {c["gt"]} | {lq:.1f} | {lt:.1f}'
        err['sq'].append(lq - c['gt'])
        err['ti'].append(lt - c['gt'])
        for l in labels:
            r = res[l].get(n)
            row += f' | {r["count"] if r else "–"}'
            err[l].append((r['count'] - c['gt']) if r else float('nan'))
        if synthetic:
            for l in labels:
                r = res[l].get(n)
                row += ' | ' + ('%d / %d / %d' % match(c['truth'], r['boxes']) if r else '–')
        print(row + ' |')
    mae = lambda v: sum(abs(x) for x in v) / len(v)
    mape = lambda v: 100 * sum(abs(x) / cases[n]['gt'] for x, n in zip(v, names)) / len(v)
    print('| **MAE** | | %.1f | %.1f | ' % (mae(err['sq']), mae(err['ti'])) + ' | '.join('%.1f' % mae(err[l]) for l in labels) + (' |' * (len(labels) + 1) if synthetic else ' |'))
    print('| **mean abs. %% error** | | %.1f %% | %.1f %% | ' % (mape(err['sq']), mape(err['ti'])) + ' | '.join('%.1f %%' % mape(err[l]) for l in labels) + (' |' * (len(labels) + 1) if synthetic else ' |'))
    print()


names = list(cases)
table([n for n in names if 'truth' in cases[n]], 'synthetic scene', True)
table([n for n in names if 'truth' not in cases[n] and not cases[n].get('mosaic')], 'FSC-147 image', False)
table([n for n in names if cases[n].get('mosaic')], 'FSC-147 mosaic', False)
