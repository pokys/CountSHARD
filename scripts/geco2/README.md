# GeCo2 → ONNX (Mode B, second model)

How the GeCo2 graphs the app downloads were made and measured. The app itself uses
`src/ai/geco2.ts`, `src/ai/device.ts` and the worker; the files are served from the separate
Pages site `https://pokys.github.io/geco2-bench/` (its source is `bench-repo/` here, models
added by `make_bench_inputs.py` + a copy of the `onnx3` exports). Weights and the GeCo2 code are
**not** in this folder; run everything in a throw-away Python venv outside the repo.

The shipped graphs are the lean, WebGPU-fixed export (`export_geco2.py` as it is now) at 768
and 1024 px, quantised with `run_onnx.py quant`; their sizes and SHA-256 are in
`src/ai/modelStore.ts`. `cpu-bench.ts` prints the app's CPU benchmark (used to calibrate
`WASM_768_S_PER_BENCH_MS` in `src/ai/device.ts`).

## Inputs

| What | Source | Licence |
|---|---|---|
| GeCo2 code | <https://github.com/jerpelhan/GECO2> (`main`, commit `67d27e8`, 2026-10-12) | MIT |
| GeCo2 weights `CNTQG_multitrain_ca44.pth` (350,785,678 bytes, SHA-256 `7841dcfd2d0120e65b40a5c6dd1dfcb014df350a34f320d2fd000dc5f4c005e7`) | huggingface.co/datasets/jerpelhan/geco2-assets | see "Licences" |
| SAM2 Hiera B+ `sam2_hiera_base_plus.pt` (323,493,298 bytes, SHA-256 `d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071`) — only for the PyTorch reference (GeCo2's mask decoder) | dl.fbaipublicfiles.com/segment_anything_2/072824/ | Apache 2.0 |
| Tools | Python 3.14, torch 2.14.1+cpu, torchvision 0.29.1, onnx 1.23.2, onnxruntime 1.31.0, hydra-core, scikit-image, tqdm; onnxruntime-web 1.30 (Node 24 and Chromium) | — |

The GeCo2 checkpoint holds 75.3 M parameters (Hiera B+ trunk 68.7 M, neck 0.4 M, counting
head 6.2 M); most of its 351 MB is optimizer state. It does **not** contain the SAM mask
decoder used for box refinement; GeCo2 loads that from the SAM2 checkpoint at construction.

## Files

- `dump-cases.ts` — writes the Mode B evaluation cases (10 synthetic scenes as PNG + true
  circles, the FSC-147 sample and its 2 × 2 mosaics) for the Python scripts:
  `npx vite-node scripts/geco2/dump-cases.ts <cases-dir>`.
- `geco2_env.py` — loads GeCo2 on CPU without its CUDA extension (Deformable-DETR
  `MSDeformAttn` via the pure-PyTorch `grid_sample` path; reference points that the original only
  builds on CUDA).
- `ref_pytorch.py` — PyTorch reference with GeCo2's own preprocessing and counting rule.
- `export_geco2.py` — the three ONNX graphs + check against the original model.
- `run_onnx.py` — weight-only int8 quantisation, onnxruntime runs, detection post-processing
  in NumPy, input dumps for the browser bench.
- `geco2-probe.mjs` — memory/speed in onnxruntime-web WASM (Node), like `scripts/ai-mem-probe.mjs`.
- `bench.html` — the same graphs in the browser (WebGPU / WASM).
- `make_bench_inputs.py` + `bench-repo/` — the phone test page published as
  `pokys.github.io/geco2-bench` (also the host of the app's GeCo2 model files).
- `bench-loca.html` — one LOCA tile on the same bench (GPU-memory comparison).
- `debug_outputs.py` — exposes intermediate tensors and compares CPU/WebGPU fingerprints
  (`bench.html?debug=backbone|head`); used to find the WebGPU problems.
- `compare.py` — GeCo2 vs LOCA tables (LOCA numbers parsed from `docs/testing.md`).
- `draw.py` — draws detections for visual checks.

## Export details that matter

- Graph sizes and numbers: `docs/testing.md` ("GeCo2 feasibility").
- `ATTN_CHUNK` (512): Hiera's 3 global-attention blocks in query chunks.
- `BAND_TOKENS` (8192): windowed Hiera blocks run in bands of whole window rows (the 1024 px
  stage-1 MLP would otherwise create 117 MB tensors + GELU temporaries).
- `DEFORM_CHUNK` (8192): `grid_sample` of the deformable attention in query chunks.
- `STRIPES` (8): up2 + up3 + class/box heads computed in row stripes of the 512 × 512 output.
- Row indices of the striped bilinear upsampling are literal constants: computed in-graph they
  left an int64 `Clip`, which onnxruntime-web 1.30 WebGPU evaluates wrongly.
- `run_onnx.py quant` resolves the `Identity` nodes the exporter puts in front of weights used by
  several bands, otherwise those weights stay fp32 (84 MB instead of 71 MB).
- SAM mask-decoder refinement is not exported (≈ 4 MAE on FSC-147, 0 on synthetic).

## Licences

GeCo2 code: MIT (repository LICENSE). Backbone: SAM2 Hiera B+ (Apache 2.0). The Hugging Face
dataset `jerpelhan/geco2-assets` that hosts the checkpoint has **no licence tag** (checked via
its API, 2026-10-20), and the weights were trained on counting datasets (FSC-147 etc.) whose
images come from the web. Treat the weights' licence as unclear and ask the authors before
shipping (their README asks users to cite and notify them by e-mail).

## Commands

```bash
npx vite-node scripts/geco2/dump-cases.ts $T/cases
TORCH_HOME=$T/torch python -I scripts/geco2/ref_pytorch.py $T/code $T/CNTQG_multitrain_ca44.pth $T/cases $T/ref.json
TORCH_HOME=$T/torch SIZE=1024 python -I scripts/geco2/export_geco2.py $T/code $T/CNTQG_multitrain_ca44.pth $T/cases $T/onnx
python -I scripts/geco2/run_onnx.py quant $T/onnx 1024
python -I scripts/geco2/run_onnx.py run $T/onnx 1024 w8 $T/cases $T/w8.json 4
node scripts/geco2/geco2-probe.mjs $T/onnx 1024 w8 disabled
```

(`$T/torch/hub/checkpoints/sam2_hiera_base_plus.pt` must exist; GeCo2 fetches it through
`torch.hub` otherwise.)
