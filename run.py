from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from models.model import DefendSRLite


def _to_grayscale_2d(arr: np.ndarray) -> np.ndarray:
    """Convert supported grayscale-like arrays to (H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[0] == 1:
            return arr[0]
        # Defensive fallback if an unexpected multi-channel array appears.
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1)

    raise ValueError(f"Expected grayscale array of shape (H,W) or (H,W,1); got {arr.shape}")


def _sanitize(arr: np.ndarray) -> np.ndarray:
    arr = _to_grayscale_2d(arr).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.5, neginf=-0.5)
    return np.ascontiguousarray(arr)


def load_model(device: torch.device) -> DefendSRLite:
    model = DefendSRLite(channels=24, blocks=4)
    weight_path = Path(__file__).resolve().parent / "models" / "defend_sr_lite.pt"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Required model weights are missing: {weight_path}. "
            "Keep models/defend_sr_lite.pt inside the repository."
        )

    checkpoint = torch.load(weight_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def restore_one(model: DefendSRLite, arr: np.ndarray, device: torch.device) -> np.ndarray:
    arr = _sanitize(arr)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        y = model(x)

    out = y.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    expected = (arr.shape[0] * 2, arr.shape[1] * 2)
    if out.shape != expected:
        raise RuntimeError(f"Wrong output shape {out.shape}; expected {expected}")
    if not np.isfinite(out).all():
        raise RuntimeError("Output contains NaN or Inf")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DEFEND-SR Lite: restore all grayscale .npy images with x2 super-resolution."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing degraded .npy files")
    parser.add_argument("output_dir", type=Path, help="Directory for restored .npy files")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy")
    if not files:
        raise SystemExit(f"No .npy files found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"DEFEND-SR Lite | device={device} | files={len(files)}")
    model = load_model(device)

    # GPU warm-up to avoid counting one-time kernel startup in per-image time.
    if device.type == "cuda":
        dummy = torch.zeros(1, 1, 128, 128, device=device)
        with torch.inference_mode():
            _ = model(dummy)
        torch.cuda.synchronize()

    times = []
    for idx, path in enumerate(files, 1):
        arr = np.load(path, allow_pickle=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        restored = restore_one(model, arr, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        out_path = args.output_dir / path.name
        np.save(out_path, restored, allow_pickle=False)
        print(f"[{idx:04d}/{len(files):04d}] {path.name}: {arr.shape} -> {restored.shape} | {elapsed*1000:.2f} ms")

    finite_times = [t for t in times if math.isfinite(t)]
    if finite_times:
        print(f"Done. Mean inference: {np.mean(finite_times)*1000:.2f} ms/image")
    print(f"Saved restored arrays to: {args.output_dir}")


if __name__ == "__main__":
    main()
