# Plan: own detector for steel products seen end-on (YOLO)

Status: plan, 2026-10-25. Nothing trained yet.

## Goal

A third optional model in the AI model switch (*Automaticky / LOCA / GeCo2 / vlastní model*)
that detects **individual end faces** of steel products and tells their **type**:

| Class (v1) | Czech | Notes |
|---|---|---|
| `round` | trubky, tyče, kulatina | hollow and solid round together in v1 |
| `square` | jekly čtvercové | |
| `rectangular` | jekly obdélníkové | |
| `h_beam` | HEA/HEB/IPE ("hépéčka", I) | H and I together in v1 |
| `l_angle` | úhelníky L ("elka") | **no usable public data**, see below |
| `u_channel` | U profily | **no usable public data**, see below |

Detections become ordinary editable objects (numbered boxes, tap to add/remove, counting area,
undo), grouped by class and size — the GeCo2 detection path in the app already does this.

## Decisions

- **Model family: Ultralytics YOLO, newest generation (YOLO26)**, sizes `n` (phones) and `s`
  (computers). NMS-free end-to-end export, ONNX for onnxruntime-web (already in the app).
- **Licence: Ultralytics is AGPL-3.0.** CountSHARD switches from MIT to **AGPL-3.0** when the model
  is integrated (owner's decision, 2026-10-25): update `LICENSE`, `package.json`, README,
  `THIRD_PARTY_NOTICES.md`. The trained weights are distributed with the app under AGPL-3.0.
- **Datasets: Roboflow Universe, CC BY 4.0 only** (attribution in `THIRD_PARTY_NOTICES.md` and in
  a `DATASETS.md` next to the weights). Anything with NC/ND/unknown licence is excluded.
- **Owner's corrected photos are the most valuable data** and part of them is a held-out test set
  that is never used for training.

## Dataset shortlist (Roboflow Universe, checked 2026-10-25)

All listed as *Object Detection*, licence **CC BY 4.0** on the project page. Image counts are the
project totals. **Content and label quality are not verified yet** (preview needs sign-in) —
step 1 below checks every one before use.

| Project | Images | Classes seen in search | Use for |
|---|---|---|---|
| [hiral/pipe-counting-4rbib](https://universe.roboflow.com/hiral/pipe-counting-4rbib) | 9,472 | `circle`, `R 20 40 …`, `S 50 50 …` (sizes of rectangular/square tubes) | round, square, rectangular — **main source** |
| [home-1ip0r/count_tube](https://universe.roboflow.com/home-1ip0r/count_tube) | 978 | tube | round |
| [steel-rod-counting/steel-rod-counting](https://universe.roboflow.com/steel-rod-counting/steel-rod-counting) | 793 | 8 mm, 16 mm, 32 mm | round (solid rods) |
| [ztai/count-steels-3](https://universe.roboflow.com/ztai/count-steels-3) | 757 | (to check) | round |
| [thesis-hfm2k/steel-pipe-size-detection-8b7dm](https://universe.roboflow.com/thesis-hfm2k/steel-pipe-size-detection-8b7dm) | 470 | 1½", 1", 2", 3" | round |
| [mythesis-hhdfg/steel-rectangular-tube](https://universe.roboflow.com/mythesis-hhdfg/steel-rectangular-tube) | 463 | Rectangular hollow steel bar | rectangular |
| [project-qoavl/h-beam-5bkp6](https://universe.roboflow.com/project-qoavl/h-beam-5bkp6) | 449 | H_steel | h_beam (check it is end-on) |
| [ztai/round-steels-1](https://universe.roboflow.com/ztai/round-steels-1) | 399 | (to check) | round |
| [john-tai-fa0rl/pipe-detector-5](https://universe.roboflow.com/john-tai-fa0rl/pipe-detector-5) | 274 | (to check) | round |
| [satyams-workspace-urdlm/steel-pipe-63hd3](https://universe.roboflow.com/satyams-workspace-urdlm/steel-pipe-63hd3) | 240 | steel pipe | round |
| [celebal-technology-kubjx/counting-pipe](https://universe.roboflow.com/celebal-technology-kubjx/counting-pipe) | 197 | pipe | round (likely the same images as nikitas-celebal-workspace/pipe-counting-3) |
| [isbcapstone-y8t7f/pipe-detection-and-counting](https://universe.roboflow.com/isbcapstone-y8t7f/pipe-detection-and-counting) | 144 | circular, square, rectangular, oval pipe | round, square, rectangular |
| [abhijeet-t8xgm/square-pipes](https://universe.roboflow.com/abhijeet-t8xgm/square-pipes) | 143 | object | square |
| [steel-rebars/steel-rebar-counting-system](https://universe.roboflow.com/steel-rebars/steel-rebar-counting-system) | 131 | rebar 10–36 mm | round (rebar) |
| [my-workspace-gdkcu/count-square-steel](https://universe.roboflow.com/my-workspace-gdkcu/count-square-steel) | 76 | SquareSteel | square |
| [personal-ienlf/steel-beam-counting](https://universe.roboflow.com/personal-ienlf/steel-beam-counting) | 52 | steel-beam | h_beam |
| [mat-r5pdt/material-ymkye](https://universe.roboflow.com/mat-r5pdt/material-ymkye) | 39 | CHANNEL, EL-SQUARE, H-BEAM… | u_channel, l_angle (tiny) |

**Gap: L angles and U channels.** Searches for angle steel, angle iron, channel and steel
profiles found nothing usable beyond the 39-image set above. These two classes need the owner's
own photos (target ≥ 150 annotated images each, ideally several hundred objects per class) or
they stay out of v1. Synthetic renders can supplement but not replace real photos.

## Steps

1. **Download and audit** (½–1 day). Export each project in YOLO format via the Roboflow API with
   the owner's free Roboflow account (API key in an environment variable, never in the repo).
   Per dataset: look at ~30 random images with labels; drop sets that are not end-on views, have
   sloppy boxes, or label bundles instead of single pieces. Remap classes to the v1 taxonomy.
   Deduplicate across sets (perceptual hash) — many Universe projects are forks of each other.
2. **Own data** (owner). Photograph real stacks, count and correct them in CountSHARD, export JSON.
   A converter `scripts/yolo/countshard-to-yolo.mjs` turns CountSHARD projects into YOLO labels
   (circles → boxes; GeCo2/manual boxes as they are; class chosen per size group). Split: ≥ 20
   photos held out as the test set; the rest goes to training.
3. **Train** (hours per run). Primary: Google Colab (free T4). Recipe to start from:
   `yolo26s.pt`, `imgsz=1024` (end faces in large bundles are small), `epochs=150`,
   `patience=30`, default mosaic/HSV augmentation, `close_mosaic=15`, `max_det=1000`, flips on.
   Train/val split by source dataset so near-duplicates do not leak into validation. Then `yolo26n`
   with the same data for phones.
   Local alternative on the GTX 1050 Ti (4 GB, Pascal): update the NVIDIA driver first (current
   456.71 is from 2020 and too old for CUDA 12); use PyTorch's legacy CUDA 12.6 build, which still
   includes Pascal, and confirm `torch.cuda.get_arch_list()` contains `sm_61`; small batches, no
   AMP (Pascal has no tensor cores). Expect several times slower than Colab.
4. **Evaluate** (½ day) on the owner's held-out photos only: count error per photo (absolute and
   %), precision/recall at IoU 0.5, per-class confusion (square vs rectangular!), and the same
   photos through the current circle detector and GeCo2 for comparison. Report honestly; no
   numbers from the validation split in the README.
5. **Export and measure in the browser** (½ day). ONNX end-to-end export, optional int8; check
   size (target `n` ≤ 10 MB), time per photo with WebGPU/WASM on a computer, iPhone and a Galaxy A,
   and peak memory. Very dense bundles (> ~300 pieces) or tiny ends: tile the photo (overlapping
   1024 px tiles, merge duplicates) — the app already does tiling for LOCA.
6. **Integrate** (1–2 days). Third model in the switch, downloaded on first use and cached for
   offline use; detections through the existing GeCo2 detection path; groups per class (and size
   within `round`); model and classes recorded in exports. Licence switch to AGPL-3.0 in the same
   change. Tests for the class mapping, converter and post-processing.

## What is needed from the owner

- A free Roboflow account (for the dataset export API key).
- 20–50 corrected photos of real stacks for the test set; more for training.
- Photos of L angles and U channels if those classes should be in v1.
- Either a Google account for Colab, or the NVIDIA driver update on the local PC.

## Risks

- Public datasets are small, noisy and overlapping; the 9.5k-image set dominates and may bias the
  model to its own camera/lighting. Mitigation: per-source validation and the owner's test set.
- Square vs rectangular confusion for nearly square tubes; may need merging into one class.
- Domain gap to the owner's warehouse (rust, paint marks, labels, perspective).
- AGPL-3.0 applies to the whole app once the model ships.
