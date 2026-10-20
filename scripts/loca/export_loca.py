"""Export LOCA few-shot to ONNX as two graphs (features + head) and verify vs PyTorch.

usage: python -I export_loca.py <loca_src_dir> <weights.pt> <out_dir>
"""
import sys, os, time, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

loca_dir, weights, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, loca_dir)

import torchvision.models as tvm
_orig_resnet50 = tvm.resnet50
def _resnet50(pretrained=False, **kw):  # never download ImageNet/SwAV weights: the checkpoint has them
    return _orig_resnet50(weights=None, **kw)
tvm.resnet50 = _resnet50

from models.loca import LOCA  # noqa: E402
from models import positional_encoding as _pe  # noqa: E402

# ONNX CumSum does not accept bool; LOCA cumsums a bool mask. Same maths on a float mask.
def _1d_pos_enc(self, mask, dim):
    temp = torch.arange(self.emb_dim // 2).float().to(mask.device)
    temp = self.temperature ** (2 * (temp.div(2, rounding_mode='floor')) / self.emb_dim)
    enc = (1.0 - mask.float()).cumsum(dim).unsqueeze(-1) / temp
    return torch.stack([enc[..., 0::2].sin(), enc[..., 1::2].cos()], dim=-1).flatten(-2)
_pe.PositionalEncodingsFixed._1d_pos_enc = _1d_pos_enc

torch.manual_seed(0)
model = LOCA(
    image_size=512, num_encoder_layers=3, num_ope_iterative_steps=3, num_objects=3,
    emb_dim=256, num_heads=8, kernel_dim=3, backbone_name='resnet50', swav_backbone=False,
    train_backbone=False, reduction=8, dropout=0.0, layer_norm_eps=1e-5, mlp_factor=8,
    norm_first=True, activation=nn.GELU, norm=True, zero_shot=False,
)
ckpt = torch.load(weights, map_location='cpu', weights_only=True)
sd = ckpt['model'] if 'model' in ckpt else ckpt
sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval()


class ChunkedSelfAttention(nn.Module):
    """Same maths as nn.MultiheadAttention (batch_first=False, no masks, eval), but `chunk` queries
    at a time, so the largest intermediate is heads × chunk × S floats instead of heads × L × S
    (8 × 4096² × 4 B = 512 MB at 64 × 64 tokens, which phones cannot allocate)."""
    def __init__(self, mha: nn.MultiheadAttention, chunk: int):
        super().__init__()
        self.mha, self.chunk = mha, chunk
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, **kw):
        assert attn_mask is None and key_padding_mask is None
        m = self.mha
        E, H = m.embed_dim, m.num_heads
        d = E // H
        W, b = m.in_proj_weight, m.in_proj_bias
        L, N, _ = query.shape
        S = key.shape[0]
        assert N == 1
        q = F.linear(query, W[:E], b[:E]).reshape(L, H, d).transpose(0, 1) * (d ** -0.5)
        k = F.linear(key, W[E:2 * E], b[E:2 * E]).reshape(S, H, d).transpose(0, 1)
        v = F.linear(value, W[2 * E:], b[2 * E:]).reshape(S, H, d).transpose(0, 1)
        kt = k.transpose(1, 2)  # (H, d, S)
        parts = [torch.softmax(q[:, s:s + self.chunk] @ kt, dim=-1) @ v for s in range(0, L, self.chunk)]
        # Pairwise concatenation: onnxruntime-web 1.30's WebGPU Concat returns wrong data with
        # 8 or more inputs (measured: exact with 4 inputs or a pairwise tree).
        while len(parts) > 1:
            parts = [torch.cat(parts[i:i + 2], 1) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
        o = parts[0].transpose(0, 1).reshape(L, N, E)
        return F.linear(o, m.out_proj.weight, m.out_proj.bias), None


ATTN_CHUNK = int(os.environ.get('ATTN_CHUNK', '0'))
original_encoder_attn = [layer.self_attn for layer in model.encoder.layers]
if ATTN_CHUNK:
    # Only the image encoder attends over all 64 × 64 tokens; the OPE attention in the head is small.
    with torch.no_grad():
        x = torch.randn(4096, 1, 256)
        ref_attn = original_encoder_attn[0](x, x, x)[0]
        got_attn = ChunkedSelfAttention(original_encoder_attn[0], ATTN_CHUNK)(x, x, x)[0]
        print(f'chunked attention ({ATTN_CHUNK} queries) vs nn.MultiheadAttention: max abs diff {(ref_attn - got_attn).abs().max().item():.2e}')
    for layer in model.encoder.layers:
        layer.self_attn = ChunkedSelfAttention(layer.self_attn, ATTN_CHUNK)
print('loaded', sum(p.numel() for p in model.parameters()) / 1e6, 'M params')


class Features(nn.Module):
    """image [1,3,512,512] (ImageNet-normalised) -> encoded features [1,256,64,64]."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        m = self.m
        src = m.input_proj(m.backbone(x))
        bs, c, h, w = src.size()
        pos = m.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)
        f = m.encoder(src, pos, src_key_padding_mask=None, src_mask=None)
        return f.permute(1, 2, 0).reshape(bs, m.emb_dim, h, w)


class Head(nn.Module):
    """features of the tile to count + features of the exemplar tile + 3 boxes (x1,y1,x2,y2 in
    512 px of the exemplar tile) -> density [1,1,512,512]. With tile == exemplar tile this is
    exactly LOCA.forward."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, f_tile, f_ex, bboxes):
        m = self.m
        bs, _, h, w = f_tile.size()
        pos = m.pos_emb(bs, h, w, f_tile.device).flatten(2).permute(2, 0, 1)
        protos = m.ope(f_ex, pos, bboxes)[-1]
        n = m.num_objects
        protos = protos.permute(1, 0, 2).reshape(bs, n, m.kernel_dim, m.kernel_dim, -1) \
            .permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]
        resp = F.conv2d(
            torch.cat([f_tile for _ in range(n)], dim=1).flatten(0, 1).unsqueeze(0),
            protos, bias=None, padding=m.kernel_dim // 2, groups=protos.size(0),
        ).view(bs, n, m.emb_dim, h, w).max(dim=1)[0]
        return m.regression_head(resp)


feat, head = Features(model).eval(), Head(model).eval()

# --- inputs: a few FSC-147-like test images (squashed to 512 like LOCA's own eval) ---
from PIL import Image
fsc = os.environ.get('FSC_DIR')
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
samples = []
if fsc:
    ann = json.load(open(os.path.join(fsc, 'annotation_FSC147_384.json')))
    names = sorted(os.listdir(os.path.join(fsc, 'images')))[:6]
    for n in names:
        im = Image.open(os.path.join(fsc, 'images', n)).convert('RGB'); W, H = im.size
        a = (np.asarray(im.resize((512, 512), Image.BILINEAR), np.float32) / 255 - mean) / std
        b = np.array(ann[n]['box_examples_coordinates'][:3], np.float32)[:, [0, 2], :].reshape(-1, 4)
        b = b / np.array([W, H, W, H], np.float32) * 512
        samples.append((n, torch.from_numpy(a.transpose(2, 0, 1)[None].copy()), torch.from_numpy(b[None]), len(ann[n]['points'])))
else:
    samples.append(('random', torch.randn(1, 3, 512, 512), torch.tensor([[[10, 10, 60, 50], [100, 100, 140, 150], [200, 210, 230, 260]]], dtype=torch.float32), -1))

# --- PyTorch reference ---
ref = {}
with torch.no_grad():
    for n, x, b, gt in samples:
        t = time.time(); d, _ = model(x, b); dt = time.time() - t
        ref[n] = d.numpy()
        # sanity: split graphs == original model
        d2 = head(feat(x), feat(x), b).numpy()
        print(f'{n}: torch count {d.sum().item():.2f} (gt {gt}), split max abs diff {np.abs(d2 - ref[n]).max():.2e}, {dt*1000:.0f} ms')

# --- export ---
x0, b0 = samples[0][1], samples[0][2]
with torch.no_grad():
    f0 = feat(x0)
kw = dict(opset_version=17, do_constant_folding=True)
use_dynamo = os.environ.get('DYNAMO') == '1'
torch.onnx.export(feat, (x0,), os.path.join(out, 'loca_features_fp32.onnx'), input_names=['image'], output_names=['features'], dynamo=use_dynamo, **kw)
torch.onnx.export(head, (f0, f0, b0), os.path.join(out, 'loca_head_fp32.onnx'), input_names=['features', 'exemplar_features', 'bboxes'], output_names=['density'], dynamo=use_dynamo, **kw)
import onnx  # noqa: E402
for part in ('features', 'head'):
    g = onnx.load(os.path.join(out, f'loca_{part}_fp32.onnx')).graph
    widest = max((len(n.input) for n in g.node if n.op_type == 'Concat'), default=0)
    # onnxruntime-web 1.30 WebGPU: Concat with >= 8 inputs gives wrong results.
    assert widest < 8, f'{part}: Concat with {widest} inputs'
print('exported')
np.savez(os.path.join(out, 'reference.npz'), **{k: v for k, v in ref.items()})
torch.save([(n, x, b, gt) for n, x, b, gt in samples], os.path.join(out, 'samples.pt'))
