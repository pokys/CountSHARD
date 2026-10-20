"""Loads GeCo2 (github.com/jerpelhan/GECO2) on CPU without its CUDA extension.

- `models.ops.modules.ms_deform_attn` is provided here: Deformable-DETR's MSDeformAttn with the
  pure-PyTorch `grid_sample` path (what the ONNX export needs), optionally in query chunks.
- SAM2 Hiera B+ weights are looked up by torch.hub in $TORCH_HOME/hub/checkpoints (the GeCo2
  checkpoint has the backbone, but GeCo2's MaskProcessor takes the SAM mask decoder from there).

Usage:  env = load(code_dir, ckpt)  ->  env.model (CNT, eval), env.box_ops, env.ops
"""
import math
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

# Number of queries per grid_sample call in MSDeformAttn (0 = all at once). Same maths either way.
DEFORM_CHUNK = int(os.environ.get('DEFORM_CHUNK', '0'))


def deform_core(value, shapes, sampling_locations, attention_weights):
    """ms_deform_attn_core_pytorch for one level, chunked over queries."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    assert L_ == 1
    H_, W_ = shapes[0]
    value_l = value.flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
    grids = (2 * sampling_locations - 1)[:, :, :, 0].transpose(1, 2).flatten(0, 1)  # N*M, Lq, P, 2
    aw = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    step = DEFORM_CHUNK if DEFORM_CHUNK > 0 else Lq_
    outs = []
    for s in range(0, Lq_, step):
        sv = F.grid_sample(value_l, grids[:, s:s + step], mode='bilinear', padding_mode='zeros', align_corners=False)
        outs.append((sv * aw[:, :, s:s + step]).sum(-1))  # N*M, D, chunk
    out = outs[0]
    while len(outs) > 1:  # pairwise concat (ORT-Web WebGPU: Concat with >= 8 inputs is wrong)
        outs = [torch.cat(outs[i:i + 2], dim=2) if i + 1 < len(outs) else outs[i] for i in range(0, len(outs), 2)]
        out = outs[0]
    return out.view(N_, M_ * D_, Lq_).transpose(1, 2)


class MSDeformAttn(nn.Module):
    """Deformable-DETR MSDeformAttn (same parameters), CPU/ONNX path."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        shapes = [(int(h), int(w)) for h, w in input_spatial_shapes.tolist()]
        value = self.value_proj(input_flatten).view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        off = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        aw = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        aw = F.softmax(aw, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        norm = torch.tensor([[w, h] for h, w in shapes], dtype=query.dtype)
        loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        return self.output_proj(deform_core(value, shapes, loc, aw))


def install_shims(code_dir):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    for name in ['models.ops', 'models.ops.modules']:
        sys.modules.setdefault(name, types.ModuleType(name))
    m = types.ModuleType('models.ops.modules.ms_deform_attn')
    m.MSDeformAttn = MSDeformAttn
    sys.modules['models.ops.modules.ms_deform_attn'] = m


def load(code_dir, ckpt):
    install_shims(code_dir)
    cwd = os.getcwd()
    os.chdir(code_dir)  # hydra config module + relative config name
    try:
        from models.counter_infer import build_model
        from utils.arg_parser import get_argparser
        args = get_argparser().parse_args([])
        args.zero_shot = True
        model = build_model(args)
    finally:
        os.chdir(cwd)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)['model']
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Expected missing: only the SAM mask processor (loaded from SAM2 in its constructor).
    bad = [k for k in missing if not k.startswith('sam_mask.')]
    assert not bad, bad
    model.eval()
    # C_base only builds its reference points when moved to another device (CUDA); do it for CPU.
    af = model.adapt_features
    af.reference_points = af.get_reference_points(af.spatial_shapes, af.valid_ratios)
    af.reference_points1 = af.get_reference_points(af.spatial_shapes1, af.valid_ratios1)
    af.reference_points2 = af.get_reference_points(af.spatial_shapes2, af.valid_ratios2)
    from utils import box_ops
    from torchvision import ops
    return types.SimpleNamespace(model=model, box_ops=box_ops, ops=ops)
