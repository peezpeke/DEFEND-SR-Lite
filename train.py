"""Reproduce the shipped prototype checkpoint using synthetic semiconductor-like data.

This is intentionally self-contained: it does not download any dataset or model.
For a serious competition run, replace the synthetic generator with KLA's paired
training data and keep the same four-term objective / Defect Guardian weighting.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.model import DefendSRLite


def draw_batch(batch: int, size: int, device: torch.device):
    """Generate simple IC/SEM-like grayscale patterns plus explicit defect masks."""
    imgs = torch.zeros(batch, 1, size, size, device=device)
    masks = torch.zeros_like(imgs)

    yy = torch.arange(size, device=device).view(1, size, 1)
    xx = torch.arange(size, device=device).view(1, 1, size)

    for b in range(batch):
        bg = random.uniform(0.02, 0.22)
        img = torch.full((size, size), bg, device=device)

        # Periodic horizontal/vertical interconnect-like structures.
        for orientation in ("v", "h"):
            if random.random() < 0.9:
                spacing = random.randint(7, 18)
                width = random.randint(1, 3)
                offset = random.randint(0, spacing - 1)
                intensity = random.uniform(0.45, 0.95)
                if orientation == "v":
                    line = (((torch.arange(size, device=device) - offset) % spacing) < width).float()
                    img = torch.maximum(img, intensity * line.unsqueeze(0).expand(size, -1))
                else:
                    line = (((torch.arange(size, device=device) - offset) % spacing) < width).float()
                    img = torch.maximum(img, intensity * line.unsqueeze(1).expand(-1, size))

        # Contacts / vias.
        for _ in range(random.randint(6, 22)):
            cx, cy = random.randrange(size), random.randrange(size)
            r = random.randint(1, 3)
            disk = ((xx[0] - cx) ** 2 + (yy[0] - cy) ** 2 <= r * r)
            img[disk] = max(float(img[disk].max()) if disk.any() else 0.0, random.uniform(0.65, 1.0))

        # Defect Guardian: create labelled breaks, bridges, protrusions / missing contacts.
        defect_mask = torch.zeros((size, size), device=device)
        for _ in range(random.randint(1, 4)):
            kind = random.choice(["break_v", "break_h", "bridge", "blob"])
            cx, cy = random.randint(5, size - 6), random.randint(5, size - 6)
            if kind == "break_v":
                w, h = random.randint(2, 4), random.randint(4, 9)
                sl = (slice(cy - h // 2, cy + h // 2 + 1), slice(cx - w // 2, cx + w // 2 + 1))
                img[sl] = bg
                defect_mask[sl] = 1.0
            elif kind == "break_h":
                w, h = random.randint(4, 9), random.randint(2, 4)
                sl = (slice(cy - h // 2, cy + h // 2 + 1), slice(cx - w // 2, cx + w // 2 + 1))
                img[sl] = bg
                defect_mask[sl] = 1.0
            elif kind == "bridge":
                w, h = random.randint(5, 11), random.randint(1, 3)
                sl = (slice(cy - h, cy + h + 1), slice(cx - w, cx + w + 1))
                img[sl] = random.uniform(0.65, 1.0)
                defect_mask[sl] = 1.0
            else:
                r = random.randint(2, 4)
                disk = ((xx[0] - cx) ** 2 + (yy[0] - cy) ** 2 <= r * r)
                img[disk] = random.uniform(0.65, 1.0)
                defect_mask[disk] = 1.0

        # Slight blur approximates an inspection point-spread function.
        img = img.unsqueeze(0).unsqueeze(0)
        img = F.avg_pool2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        imgs[b] = img[0]
        masks[b, 0] = defect_mask

    return imgs.clamp(0, 1), masks


def degrade(gt: torch.Tensor) -> torch.Tensor:
    lr = F.interpolate(gt, scale_factor=0.5, mode="area")
    speckle_sigma = torch.empty(gt.shape[0], 1, 1, 1, device=gt.device).uniform_(0.04, 0.14)
    gauss_sigma = torch.empty(gt.shape[0], 1, 1, 1, device=gt.device).uniform_(0.005, 0.04)
    lr = lr * (1.0 + speckle_sigma * torch.randn_like(lr))
    lr = lr + gauss_sigma * torch.randn_like(lr)
    return torch.clamp(lr, -0.5, 1.5)


def sobel(x: torch.Tensor):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Lightweight differentiable global-window SSIM approximation.
    mu_x = F.avg_pool2d(x, 7, 1, 3)
    mu_y = F.avg_pool2d(y, 7, 1, 3)
    sig_x = F.avg_pool2d(x * x, 7, 1, 3) - mu_x * mu_x
    sig_y = F.avg_pool2d(y * y, 7, 1, 3) - mu_y * mu_y
    sig_xy = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2) + 1e-8)
    return 1.0 - ssim.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--out", type=Path, default=Path("models/defend_sr_lite.pt"))
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefendSRLite(channels=24, blocks=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)

    for step in range(1, args.steps + 1):
        gt, defect_mask = draw_batch(args.batch_size, args.patch_size, device)
        lr = degrade(gt)
        pred, confidence = model(lr, return_confidence=True)

        l_pixel = F.l1_loss(pred, gt)
        l_ssim = ssim_loss(pred, gt)
        px, py = sobel(pred)
        gx, gy = sobel(gt)
        l_edge = F.l1_loss(px, gx) + F.l1_loss(py, gy)

        # Defect Guardian: error at procedurally inserted defect pixels receives
        # extra weight, so the network is discouraged from smoothing them away.
        weighted = (pred - gt).abs() * (1.0 + 4.0 * defect_mask)
        l_defect = weighted.mean()

        # Confidence learns to be low where restoration error is high.
        conf_target = torch.exp(-8.0 * (pred.detach() - gt).abs())
        l_conf = F.l1_loss(confidence, conf_target)

        loss = l_pixel + 0.25 * l_ssim + 0.50 * l_edge + 0.75 * l_defect + 0.05 * l_conf
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 100 == 0:
            mse = F.mse_loss(pred.detach(), gt).item()
            psnr = -10.0 * math.log10(max(mse, 1e-12))
            print(f"step={step:4d} loss={loss.item():.4f} synthetic_psnr={psnr:.2f} dB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "DefendSRLite(channels=24, blocks=4)",
            "scale": 2,
            "training": "procedural semiconductor-like patterns with defect-weighted loss",
            "seed": args.seed,
            "steps": args.steps,
        },
        args.out,
    )
    print(f"Saved checkpoint: {args.out}")


if __name__ == "__main__":
    main()
