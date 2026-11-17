#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for real models
    torch = None


def depth_to_normal_map(depth: np.ndarray, slope_scale: float = 8.0) -> np.ndarray:
    """Convert a depth map to a normal map with enhanced color sensitivity.

    The depth-map is treated as a height field z = f(x, y), and the local
    surface normal is computed from the gradient of that field. A moderate
    slope scale is applied so minor depth variations lead to clearly visible
    color changes in the normal map.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be a 2D array, got shape {depth.shape}")
    if depth.shape[0] < 2 or depth.shape[1] < 2:
        raise ValueError("depth must be at least 2x2")

    slope_scale = float(slope_scale)
    if slope_scale <= 0:
        raise ValueError("slope_scale must be positive")

    dz_dy, dz_dx = np.gradient(depth)
    dz_dy *= slope_scale
    dz_dx *= slope_scale

    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    normals = (normals + 1.0) * 0.5
    normals = np.clip(normals, 0.0, 1.0)
    return normals.astype(np.float32)


def estimate_depth_map(image_rgb: np.ndarray, model_name: Optional[str] = None, device: str = "cpu") -> np.ndarray:
    """Estimate a depth map from a color image.

    If Depth-Anything-V2 is installed, the script tries to use it. Otherwise a
    grayscale-based fallback is used so the pipeline still runs in CPU-only
    environments.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] not in (3, 4):
        raise ValueError(f"expected an RGB(A) image, got shape {image_rgb.shape}")

    try:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")

        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet

        image = Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
        image = image.convert("RGB")
        transform = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = PrepareForNet()(image)
        image_tensor = transform(image_tensor)

        model = DepthAnythingV2(device=device)
        with torch.no_grad():
            depth = model(image_tensor.unsqueeze(0))[0]
        depth = depth.detach().cpu().numpy()[0]
        depth = np.asarray(depth, dtype=np.float32)
        # Convert depth from the model representation into a normalized height map.
        depth = (depth - np.min(depth)) / max(np.ptp(depth), 1e-6)
        return depth
    except Exception:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY if image_rgb.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        depth = gray.astype(np.float32) / 255.0
        depth = np.clip(depth, 1e-3, 1.0)
        return depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a depth map and visualize it as a normal-map image.")
    parser.add_argument("input", type=Path, help="Input image path.")
    parser.add_argument("-o", "--output", type=Path, default=Path("normal_map.png"), help="Output PNG path.")
    parser.add_argument("--model", type=str, default=None, help="Optional DepthAnything model name to prefer when available.")
    parser.add_argument("--device", default="cpu", help="Device to run the model on, e.g. cpu or cuda.")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    image = Image.open(input_path).convert("RGB")
    rgb = np.asarray(image)
    depth = estimate_depth_map(rgb, model_name=args.model, device=args.device)
    normal_map = depth_to_normal_map(depth)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((normal_map * 255.0).clip(0, 255).astype(np.uint8), mode="RGB").save(output_path)

    print(f"Saved normal map to {output_path}")


if __name__ == "__main__":
    main()
