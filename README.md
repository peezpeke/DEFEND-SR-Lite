# DEFEND-SR Lite

Evidence-preserving x2 restoration prototype for degraded grayscale semiconductor inspection images.

## What this repository does

`run.py` reads every `.npy` file in an input directory, restores it, upsamples it by **2×**, and writes one `.npy` output with the **same filename** to the output directory.

The shipped prototype uses a compact PyTorch residual network with:

- joint denoising + x2 super-resolution,
- a bicubic skip path for stable reconstruction,
- residual detail prediction,
- synthetic defect-aware training (Defect Guardian weighting), and
- a lightweight confidence head used during training.

**Important:** the included checkpoint is a self-contained hackathon prototype trained on procedurally generated semiconductor-like patterns. It does not claim to be a KLA-trained production model. If the official paired training dataset is available, retrain `train.py` on that dataset for competitive image quality.

## Required repository layout

```text
DEFEND-SR-Lite/
├── run.py
├── train.py
├── requirements.txt
├── README.md
└── models/
    ├── __init__.py
    ├── model.py
    └── defend_sr_lite.pt
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The repository does not download weights or call any API at inference time.

## Required inference command

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py input output
```

### Input

- `.npy` files
- grayscale arrays shaped `(H, W)` or `(H, W, 1)`
- values may contain moderate speckle overshoot outside `[0,1]`

### Output

For each input file, `run.py` writes an output with:

- the same filename,
- grayscale shape `(2H, 2W)`,
- `float32` dtype,
- values clipped to `[0,1]`,
- no NaN or Inf values.

Examples:

- `128×128 -> 256×256`
- `256×256 -> 512×512`

## GPU / offline behavior

If CUDA is available, PyTorch automatically uses the NVIDIA GPU. Otherwise the script falls back to CPU. The model checkpoint is stored locally in `models/defend_sr_lite.pt`; inference does not require internet access, API keys, user interaction, or extra downloads.

## Reproducing the shipped prototype checkpoint

```bash
python train.py --steps 1000 --batch-size 8 --out models/defend_sr_lite.pt
```

This training script is intentionally self-contained and uses synthetic semiconductor-like line/contact patterns with procedurally inserted defects. The defect pixels receive extra restoration-loss weight so the model is explicitly discouraged from smoothing them away.

## Final submission sanity check

Before uploading, test with a clean temporary folder:

```bash
python run.py input output
```

Then verify that the output directory contains one `.npy` file for every input file, each with the same filename and twice the height/width.

## Attribution / research basis

The high-level restoration design is inspired by efficient residual image-restoration and super-resolution literature, including NAFNet-style efficiency principles, while this repository contains a small original prototype implementation rather than copied third-party model code.
