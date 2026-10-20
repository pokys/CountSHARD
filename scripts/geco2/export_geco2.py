"""Export GeCo2 to ONNX for onnxruntime-web as three graphs and verify them against PyTorch.

  geco2-backbone : image [1,3,S,S] (ImageNet-normalised, padded) -> feat [1,256,S/16,S/16],
                   fpn1 [1,256,S/8,S/8], fpn0 [1,256,S/4,S/4]      (SAM2 Hiera B+ + FPN neck)
  geco2-exemplars: feat, fpn1, fpn0, boxes [1,K,4] (x1,y1,x2,y2 in model px)
                   -> proto0 / proto1 / proto2 [1,2K,256]          (RoIAlign 1x1 + shape MLP)
  geco2-head     : feat, fpn1, fpn0, proto0..2 -> objectness [1,1,S/2,S/2], ltrb [1,4,S/2,S/2]
                   (prototype attention, deformable attention, upsampling, class/box heads)

Detections (peaks of the 3x3-max-pooled objectness, boxes from ltrb, NMS) are computed
outside the graph. GeCo2's SAM mask-decoder box refinement is not exported (see README).

Same maths as GeCo2's CNT.forward, rearranged so no intermediate is huge:
- Hiera global attention (3 blocks over (S/16)^2 tokens) in ATTN_CHUNK-query chunks.
- Deformable attention (MSDeformAttn, pure-PyTorch grid_sample path) in DEFORM_CHUNK-query chunks.
- The last upsampling layer + class/box heads run in STRIPES row stripes (the full
  256 x S/2 x S/2 map would be 268 MB at S = 1024); its bilinear rows are gathered explicitly.
- Hiera's bicubic window-position embedding as two small matrix products instead of a folded
  29 MB constant; sine position encodings computed in-graph from per-row / per-column parts.
- Concats are pairwise (onnxruntime-web 1.30 WebGPU: Concat with >= 8 inputs is wrong).

Usage: SIZE=1024 python -I export_geco2.py <geco2-code> <ckpt> <cases-dir> <out-dir> [n-samples]
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIZE = int(os.environ.get('SIZE', '1024'))
ATTN_CHUNK = int(os.environ.get('ATTN_CHUNK', '512'))
DEFORM_CHUNK = int(os.environ.get('DEFORM_CHUNK', '8192'))
STRIPES = int(os.environ.get('STRIPES', '8'))
os.environ['DEFORM_CHUNK'] = '0'  # the reference model runs unchunked; export modules chunk explicitly
import geco2_env  # noqa: E402
from ref_pytorch import load_image, preprocess  # noqa: E402


def pairwise_cat(parts, dim):
    while len(parts) > 1:
        parts = [torch.cat(parts[i:i + 2], dim) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return parts[0]


def attention(q, k, v, chunk):
    """q, k, v [B, heads, N, d] -> [B, heads, N, d]; softmax(q k^T / sqrt d) v, chunked over queries."""
    q = q * (q.shape[-1] ** -0.5)
    kt = k.transpose(-1, -2)
    n = q.shape[2]
    if chunk <= 0 or n <= chunk:
        return torch.softmax(q @ kt, -1) @ v
    return pairwise_cat([torch.softmax(q[:, :, s:s + chunk] @ kt, -1) @ v for s in range(0, n, chunk)], 2)


# ---------------------------------------------------------------- reference (original code)
code, ckpt, cases_dir, out = sys.argv[1:5]
n_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 3
os.makedirs(out, exist_ok=True)
env = geco2_env.load(code, ckpt)
model = env.model
cases = json.load(open(os.path.join(cases_dir, 'cases.json'), encoding='utf8'))
pick = [c for c in cases if c['name'].startswith(('4951 ', '2916', 'mixed-4', '5925', 'uniform-tubes', '6847'))][:n_samples]


@torch.no_grad()
def reference_maps(x, bx):
    """GeCo2 CNT.forward up to objectness + ltrb (before boxes_with_scores / SAM refinement)."""
    m = model
    feats = m.backbone(x)
    src = feats['vision_features']
    bs, c, w, h = src.shape
    red = x.shape[-1] / w
    k = bx.size(1)
    rois = torch.cat([torch.zeros(k, 1), bx.flatten(0, 1)], 1)
    from torchvision.ops import roi_align
    ex = roi_align(src, rois, 1, 1.0 / red, aligned=True).reshape(bs, k, 256)
    ex1 = roi_align(feats['backbone_fpn'][0], rois, 1, 4.0 / red, aligned=True).reshape(bs, k, 256)
    ex2 = roi_align(feats['backbone_fpn'][1], rois, 1, 2.0 / red, aligned=True).reshape(bs, k, 256)
    hw = torch.stack([bx[:, :, 2] - bx[:, :, 0], bx[:, :, 3] - bx[:, :, 1]], -1)
    shape = m.shape_or_objectness(hw).reshape(bs, -1, 256)
    a, _ = m.adapt_features(image_embeddings=src, image_pe=m.sam_prompt_encoder.get_dense_pe(),
                            prototype_embeddings=torch.cat([ex, shape], 1),
                            hq_features=feats['backbone_fpn'],
                            hq_prototypes=[torch.cat([ex1, shape], 1), torch.cat([ex2, shape], 1)],
                            hq_pos=feats['vision_pos_enc'])
    bs, c, w, h = a.shape
    a = a.view(bs, 256, -1).permute(0, 2, 1)
    obj = m.class_embed(a).view(bs, w, h, 1).permute(0, 3, 1, 2)
    ltrb = m.bbox_embed(a).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
    return obj, ltrb


samples = []
if SIZE == 1024:
    for c in pick:
        x, bx, s, wh = preprocess(load_image(c), c['exemplars'], SIZE, 80.0 * SIZE / 1024)
        t = time.time()
        obj, ltrb = reference_maps(x, bx)
        print(f"reference {c['name']}: {time.time() - t:.1f} s", flush=True)
        samples.append({'name': c['name'], 'x': x, 'bx': bx, 'wh': wh, 'obj': obj, 'ltrb': ltrb})
else:
    # The original C_base hard-codes 64 x 64 (S = 1024); other sizes are only checked end to end.
    for c in pick:
        x, bx, s, wh = preprocess(load_image(c), c['exemplars'], SIZE, 80.0 * SIZE / 1024)
        samples.append({'name': c['name'], 'x': x, 'bx': bx, 'wh': wh})

# ---------------------------------------------------------------- export modules
from models.backbones import hieradet  # noqa: E402


def msa_forward(self, x):
    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
    q, k, v = torch.unbind(qkv, 2)
    if self.q_pool:
        q = hieradet.do_pool(q.reshape(B, H, W, -1), self.q_pool)
        H, W = q.shape[1:3]
        q = q.reshape(B, H * W, self.num_heads, -1)
    o = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), ATTN_CHUNK)
    return self.proj(o.transpose(1, 2).reshape(B, H, W, -1))


BAND_TOKENS = int(os.environ.get('BAND_TOKENS', '8192'))
_block_forward = hieradet.MultiScaleBlock.forward


def block_forward(self, x):
    """Windowed Hiera blocks are independent per window (attention) and per token (norms, MLP),
    so a large map is processed in bands of whole window rows: same result, but the MLP
    hidden tensor (and its GELU temporaries) is BAND_TOKENS x 4C instead of H*W x 4C."""
    B, H, W, C = x.shape
    ws = self.window_size
    if BAND_TOKENS <= 0 or ws == 0 or H % ws or H * W <= BAND_TOKENS:
        return _block_forward(self, x)
    band = max(ws, (BAND_TOKENS // W) // ws * ws)
    return pairwise_cat([_block_forward(self, x[:, s:s + band]) for s in range(0, H, band)], 1)


def bicubic_matrix(n_in, n_out):
    """[n_out, n_in] with M @ v == F.interpolate(bicubic, align_corners=False) along one axis."""
    e = torch.eye(n_in, dtype=torch.float64).view(n_in, 1, n_in, 1)
    return F.interpolate(e, size=(n_out, 1), mode='bicubic', align_corners=False)[:, 0, :, 0].T.float()


class Backbone(nn.Module):
    def __init__(self, bb, size):
        super().__init__()
        self.bb = bb
        t = bb.trunk
        n = size // 4
        self.register_buffer('ry', bicubic_matrix(t.pos_embed.shape[2], n), persistent=False)
        self.register_buffer('rx', bicubic_matrix(t.pos_embed.shape[3], n), persistent=False)
        self.reps = n // t.pos_embed_window.shape[2]

    def forward(self, image):
        t = self.bb.trunk
        z = image[:, :1, :1, :1] * 0  # keeps the position embedding in-graph (not a folded 29 MB constant)
        x = t.patch_embed(image)
        pe = self.ry @ (t.pos_embed + z) @ self.rx.T
        pe = pe + (t.pos_embed_window + z).tile([1, 1, self.reps, self.reps])
        x = x + pe.permute(0, 2, 3, 1)
        outs = []
        for i, blk in enumerate(t.blocks):
            x = blk(x)
            if i == t.stage_ends[-1] or (i in t.stage_ends and t.return_interm_layers):
                outs.append(x.permute(0, 3, 1, 2))
        feats, _ = self.bb.neck(outs)
        feats = feats[:-self.bb.scalp] if self.bb.scalp > 0 else feats
        return feats[-1], feats[1], feats[0]


class Exemplars(nn.Module):
    def __init__(self, m, size):
        super().__init__()
        self.m = m
        self.red = 16.0

    def forward(self, feat, fpn1, fpn0, boxes):
        from torchvision.ops import roi_align
        k = boxes.shape[1]
        b = boxes[0]
        rois = torch.cat([b[:, :1] * 0, b], 1)
        e0 = roi_align(feat, rois, 1, 1.0 / self.red, sampling_ratio=-1, aligned=True).reshape(1, k, 256)
        e1 = roi_align(fpn0, rois, 1, 4.0 / self.red, sampling_ratio=-1, aligned=True).reshape(1, k, 256)
        e2 = roi_align(fpn1, rois, 1, 2.0 / self.red, sampling_ratio=-1, aligned=True).reshape(1, k, 256)
        hw = boxes[:, :, 2:] - boxes[:, :, :2]
        shape = self.m.shape_or_objectness(hw).reshape(1, k, 256)
        return torch.cat([e0, shape], 1), torch.cat([e1, shape], 1), torch.cat([e2, shape], 1)


def ref_points(n):
    ys, xs = torch.meshgrid(torch.linspace(0.5, n - 0.5, n), torch.linspace(0.5, n - 0.5, n), indexing='ij')
    return torch.stack((xs.reshape(-1) / n, ys.reshape(-1) / n), -1)[None, :, None]  # [1, n*n, 1, 2]


def sine_parts(n, num_pos_feats=128, temperature=10000):
    """PositionEmbeddingSine(256, normalize) for an n x n map as per-row / per-column halves."""
    e = torch.arange(1, n + 1, dtype=torch.float32)
    e = e / (e[-1:] + 1e-6) * 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    p = e[:, None] / dim_t
    p = torch.stack((p[:, 0::2].sin(), p[:, 1::2].cos()), dim=2).flatten(1)  # [n, 128]
    return p.view(1, n, 1, num_pos_feats), p.view(1, 1, n, num_pos_feats)  # y part, x part


def deform(layer, query, value_in, refp, n, chunk):
    """MSDeformAttn (1 level) with grid_sample in query chunks; query/value_in [1, n*n, 256]."""
    N, Lq, _ = query.shape
    M, P = layer.n_heads, layer.n_points
    D = layer.d_model // M
    value = layer.value_proj(value_in).view(N, n * n, M, D)
    off = layer.sampling_offsets(query).view(N, Lq, M, 1, P, 2)
    aw = torch.softmax(layer.attention_weights(query).view(N, Lq, M, P), -1)
    loc = refp[:, :, None, :, None, :] + off / float(n)
    grids = (2 * loc - 1)[:, :, :, 0].transpose(1, 2).flatten(0, 1)  # [M, Lq, P, 2]
    aw = aw.transpose(1, 2).reshape(N * M, 1, Lq, P)
    value_l = value.flatten(2).transpose(1, 2).reshape(N * M, D, n, n)
    step = chunk if chunk > 0 else Lq
    parts = [(F.grid_sample(value_l, grids[:, s:s + step], mode='bilinear', padding_mode='zeros', align_corners=False)
              * aw[:, :, s:s + step]).sum(-1) for s in range(0, Lq, step)]
    o = pairwise_cat(parts, 2).view(N, M * D, Lq).transpose(1, 2)
    return layer.output_proj(o)


class Head(nn.Module):
    def __init__(self, m, size):
        super().__init__()
        self.m = m
        n = size // 16
        self.n = n
        self.register_buffer('pe', m.sam_prompt_encoder.pe_layer((n, n)).flatten(1).T[None].contiguous(), persistent=False)
        for name, k in (('r0', n), ('r1', 4 * n), ('r2', 2 * n)):
            self.register_buffer(name, ref_points(k), persistent=False)
        for name, k in (('s1', 4 * n), ('s2', 2 * n)):
            py, px = sine_parts(k)
            self.register_buffer(name + 'y', py, persistent=False)
            self.register_buffer(name + 'x', px, persistent=False)

    def pos(self, py, px, k, z):
        return torch.cat([(py + z).expand(1, k, k, 128), (px + z).expand(1, k, k, 128)], -1).reshape(1, k * k, 256)

    @staticmethod
    def src_rows(hin, lo, hi):
        """Input rows read by UpsamplingBilinear2d(x2, align_corners) for output rows [lo, hi)."""
        # Plain Python numbers -> literal constants in the graph. (Computing the indices with torch
        # ops leaves an int64 Clip in the graph, which onnxruntime-web 1.30 WebGPU gets wrong.)
        ys = [r * (hin - 1) / (2 * hin - 1) for r in range(lo, hi)]
        i0 = [int(math.floor(y)) for y in ys]
        i1 = [min(i + 1, hin - 1) for i in i0]
        return torch.tensor(ys, dtype=torch.float64), torch.tensor(i0), torch.tensor(i1)

    def up_rows(self, layer, x, x_off, hin, lo, hi):
        """Rows [lo, hi) of UpsamplingLayer(x) = GELU(conv3x3(bilinear x2)), where x holds the
        input rows [x_off, x_off + x.shape[2]) of an hin-row map."""
        conv = layer.layer[1]
        hout = 2 * hin
        elo, ehi = max(lo - 1, 0), min(hi + 1, hout)
        ys, i0, i1 = (t.tolist() for t in self.src_rows(hin, elo, ehi))
        w = torch.tensor([y - i for y, i in zip(ys, i0)], dtype=torch.float32).view(1, 1, -1, 1)
        g0 = torch.tensor([i - x_off for i in i0])
        g1 = torch.tensor([i - x_off for i in i1])
        xr = x.index_select(2, g0) * (1 - w) + x.index_select(2, g1) * w
        xr = F.interpolate(xr, size=(ehi - elo, 2 * x.shape[3]), mode='bilinear', align_corners=True)
        xr = F.pad(xr, (0, 0, 1 if lo == 0 else 0, 1 if hi == hout else 0))
        return F.gelu(F.conv2d(xr, conv.weight, conv.bias, padding=(0, 1)))

    def tail(self, src1, l1):
        """up2 (+ l1) and up3 + class/box heads in row stripes of the final map; src1 = up1(src) + l2."""
        m = self.m
        af = m.adapt_features
        h2 = src1.shape[2]          # up2 input rows (S/8)
        h3 = 2 * h2                 # up3 input rows (S/4)
        hout = 2 * h3               # output rows (S/2)
        per = hout // STRIPES
        cw = [l.weight[:, :, None, None] for l in m.bbox_embed.layers]
        cb = [l.bias for l in m.bbox_embed.layers]
        lin = m.class_embed[0]
        objs, boxes = [], []
        for s in range(STRIPES):
            a, b = s * per, (s + 1) * per
            _, i0, i1 = self.src_rows(h3, max(a - 1, 0), min(b + 1, hout))
            r0, r1 = int(i0.min()), int(i1.max()) + 1
            y2 = self.up_rows(af.up2, src1, 0, h2, r0, r1) + l1[:, :, r0:r1]
            y = self.up_rows(af.up3, y2, r0, h3, a, b)
            objs.append(F.leaky_relu(F.conv2d(y, lin.weight[:, :, None, None], lin.bias), 0.01))
            h = F.relu(F.conv2d(y, cw[0], cb[0]))
            h = F.relu(F.conv2d(h, cw[1], cb[1]))
            boxes.append(torch.sigmoid(F.conv2d(h, cw[2], cb[2])))
        return pairwise_cat(objs, 2), pairwise_cat(boxes, 2)

    def forward(self, feat, fpn1, fpn0, p0, p1, p2):
        af = self.m.adapt_features
        n = self.n
        z = feat[:, :1, :1, :1].reshape(1, 1, 1, 1) * 0
        src = feat.flatten(2).permute(0, 2, 1)
        l1 = fpn0.flatten(2).permute(0, 2, 1)
        l2 = fpn1.flatten(2).permute(0, 2, 1)
        for layer in af.prototype_attention:
            src = layer(image_f=src, prototypes=p0)
        for layer in af.prototype_attention_l1:
            l1 = layer(image_f=l1, prototypes=p1)
        for layer in af.prototype_attention_l2:
            l2 = layer(image_f=l2, prototypes=p2)
        for layer in af.image_attention:
            src = deform(layer, src + self.pe, src, self.r0, n, DEFORM_CHUNK)
        pos1 = self.pos(self.s1y, self.s1x, 4 * n, z)
        for layer in af.image_attention_l1:
            l1 = deform(layer, l1 + pos1, l1, self.r1, 4 * n, DEFORM_CHUNK)
        pos2 = self.pos(self.s2y, self.s2x, 2 * n, z)
        for layer in af.image_attention_l2:
            l2 = deform(layer, l2 + pos2, l2, self.r2, 2 * n, DEFORM_CHUNK)
        src = src.transpose(1, 2).reshape(1, 256, n, n)
        l2 = l2.transpose(1, 2).reshape(1, 256, 2 * n, 2 * n)
        l1 = l1.transpose(1, 2).reshape(1, 256, 4 * n, 4 * n)
        return self.tail(af.up1(src) + l2, l1)


hieradet.MultiScaleAttention.forward = msa_forward
hieradet.MultiScaleBlock.forward = block_forward
bb, ex, hd = Backbone(model.backbone, SIZE).eval(), Exemplars(model, SIZE).eval(), Head(model, SIZE).eval()

with torch.no_grad():
    for smp in samples:
        t = time.time()
        f = bb(smp['x'])
        t1 = time.time()
        p = ex(*f, smp['bx'])
        obj, ltrb = hd(*f, *p)
        t2 = time.time()
        smp['ours'] = (obj, ltrb)
        msg = f"split modules {smp['name']}: backbone {t1 - t:.1f} s, head {t2 - t1:.1f} s"
        if 'obj' in smp:
            msg += (f"; vs original: objectness max abs {(obj - smp['obj']).abs().max():.2e} (max {smp['obj'].max():.3f}),"
                    f" ltrb max abs {(ltrb - smp['ltrb']).abs().max():.2e}")
        print(msg, flush=True)

# ---------------------------------------------------------------- export
smp = samples[0]
with torch.no_grad():
    f = bb(smp['x'])
    p = ex(*f, smp['bx'])
kw = dict(opset_version=17, do_constant_folding=True, dynamo=False)
paths = {k: os.path.join(out, f'geco2-{k}-{SIZE}-fp32.onnx') for k in ('backbone', 'exemplars', 'head')}
torch.onnx.export(bb, (smp['x'],), paths['backbone'], input_names=['image'], output_names=['feat', 'fpn1', 'fpn0'], **kw)
torch.onnx.export(ex, (*f, smp['bx']), paths['exemplars'], input_names=['feat', 'fpn1', 'fpn0', 'boxes'],
                  output_names=['proto0', 'proto1', 'proto2'], dynamic_axes={'boxes': {1: 'k'}}, **kw)
torch.onnx.export(hd, (*f, *p), paths['head'], input_names=['feat', 'fpn1', 'fpn0', 'proto0', 'proto1', 'proto2'],
                  output_names=['objectness', 'ltrb'],
                  dynamic_axes={'proto0': {1: 'k2'}, 'proto1': {1: 'k2'}, 'proto2': {1: 'k2'}}, **kw)

import onnx  # noqa: E402
from collections import Counter  # noqa: E402
for k, pth in paths.items():
    g = onnx.load(pth).graph
    widest = max((len(nd.input) for nd in g.node if nd.op_type == 'Concat'), default=0)
    assert widest < 8, f'{k}: Concat with {widest} inputs'
    ops = Counter(nd.op_type for nd in g.node)
    big = sorted(((int(np.prod(i.dims)), i.name) for i in g.initializer), reverse=True)[:3]
    print(f'{k}: {os.path.getsize(pth) / 1e6:.1f} MB, {len(g.node)} nodes, ops {dict(sorted(ops.items()))}, largest initializers {big}')

torch.save([{k: v for k, v in s.items()} for s in samples], os.path.join(out, f'samples-{SIZE}.pt'))
print('exported')
