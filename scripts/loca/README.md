# LOCA → ONNX export (Mode B model)

How `public/models/loca/loca-features-w8.onnx` and `loca-head-w8.onnx` were made. The
weights are **not** in this folder. Run this outside the repo, in a throw-away Python venv.

## Inputs

| What | Source | Licence |
|---|---|---|
| LOCA code | <https://github.com/djukicn/loca> (`main`, 2023) | MIT, © 2023 Nikola Đukić |
| Few-shot weights `loca_few_shot.pt` (148,054,575 bytes, SHA-256 `b5376f6eaf4385918ab4da42269c60b5d5f232d9ac7a98d0a56bc5fe12822521`) | Google Drive link in the LOCA README | trained from a SwAV ResNet-50 backbone, **CC BY-NC 4.0** |
| Tools used | Python 3.14, torch 2.14.1+cpu, torchvision 0.29.1, onnx 1.23.2, onnxruntime 1.31.0 | — |

## Commands

```bash
python -m venv venv
venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install onnx onnxruntime onnxscript numpy scipy pillow
# FSC_DIR (optional): folder with images/<name>.jpg and annotation_FSC147_384.json; the first
# six images (sorted) become the verification samples. Without it, one random tensor is used.
FSC_DIR=... venv/Scripts/python -I export_loca.py <loca-main> loca_few_shot.pt out
venv/Scripts/python -I quant_verify.py out
```

## What the export does

- **Two graphs instead of one.** `features`: image `[1,3,512,512]` (ImageNet-normalised) →
  encoder features `[1,256,64,64]` (ResNet-50 layers 2–4 + 3 transformer encoder layers).
  `head`: tile features + exemplar-window features + boxes `[1,3,4]` (x1, y1, x2, y2 in model
  pixels) → density `[1,1,512,512]`. With the same features in both inputs this is exactly
  `LOCA.forward(image, bboxes)` (checked: max abs. difference 0). The split lets the app
  compute exemplar prototypes once and apply them to every tile of a large photo.
- `PositionalEncodingsFixed` cumsums a bool mask, which ONNX `CumSum` rejects; the script
  replaces it with the same maths on a float mask. The ImageNet/SwAV download in LOCA's
  backbone constructor is disabled (the checkpoint already contains all weights).
- Legacy TorchScript exporter, opset 17. torchvision's RoIAlign is exported with
  `sampling_ratio = 0` (adaptive), which equals torchvision's default `-1`.
- **Weight-only int8**: every Conv/MatMul/Gemm weight with ≥ 1024 elements is stored as int8
  (symmetric, per output channel) + `DequantizeLinear`; compute stays fp32, so WebGPU and
  WASM run the same graph. 136.6 MB → 34.9 MB. Dynamic int8 quantisation
  (ConvInteger/MatMulInteger) was also tried: same size, about twice the count error.

## Verification (measured 2026-10-20, onnxruntime 1.31 CPU, 1 thread)

Six FSC-147 test images squashed to 512 × 512 as in LOCA's own evaluation, 3 annotated
exemplars each. "count" = sum of the density map.

| image | GT | PyTorch | ONNX fp32 | int8 (shipped) | int8 − PyTorch | dynamic int8 |
|---|---|---|---|---|---|---|
| 2227 | 11 | 8.53 | 8.53 | 8.27 | −0.26 | 8.14 |
| 2916 | 202 | 205.99 | 205.99 | 205.37 | −0.62 | 202.16 |
| 4943 | 65 | 63.29 | 63.29 | 62.76 | −0.53 | 61.90 |
| 4947 | 205 | 221.59 | 221.59 | 220.89 | −0.71 | 228.40 |
| 4950 | 33 | 32.63 | 32.63 | 32.60 | −0.03 | 31.88 |
| 4951 | 166 | 186.96 | 186.96 | 183.50 | −3.46 | 192.24 |

- fp32 ONNX vs PyTorch: max abs. density difference ≤ 1.8e-7, count difference < 0.001.
- Shipped int8 vs PyTorch: count difference −0.03 … −3.46 (≤ 1.9 %); per-pixel max abs.
  difference up to 6.0e-3 (38 % of that image's peak density on 4951, 5–14 % on the others) —
  the map is locally noisier, the sums stay close. The error comes from the quantised
  `features` graph (int8 head + fp32 features: ≤ 0.4 difference).
- Speed (CPU, 1 thread, native ORT): features ≈ 2.2 s, head ≈ 0.2 s per 512 × 512 tile.
  onnxruntime-web 1.30 WASM in Node: features ≈ 4.9 s, head ≈ 0.4 s.
