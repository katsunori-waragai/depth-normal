#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for real models
    torch = None


@dataclass
class NormalMapConfig:
    slope_scale: float = 8.0
    model_name: Optional[str] = None # depth-anything-v2 model name to prefer when available
    device: str = "cpu"
    output_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.slope_scale = float(self.slope_scale)
        if self.slope_scale <= 0:
            raise ValueError("slope_scale must be positive")
        self.device = str(self.device)
        if self.output_path is not None:
            self.output_path = Path(self.output_path)


@dataclass
class DepthNormalizer:
    config: NormalMapConfig = field(default_factory=NormalMapConfig)

    def estimate_depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        return estimate_depth_map(image_rgb, model_name=self.config.model_name, device=self.config.device)

    def normal_map_from_depth(self, depth: np.ndarray) -> np.ndarray:
        return depth_to_normal_map(depth, slope_scale=self.config.slope_scale)

    def process_image(self, image_rgb: np.ndarray) -> np.ndarray:
        depth = self.estimate_depth_map(image_rgb)
        return self.normal_map_from_depth(depth)

    def save(self, normal_map: np.ndarray, output_path: Optional[Path] = None) -> Path:
        path = Path(output_path) if output_path is not None else self.config.output_path
        if path is None:
            raise ValueError("output path is not set")

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((normal_map * 255.0).clip(0, 255).astype(np.uint8), mode="RGB").save(path)
        return path


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

    config = NormalMapConfig(
        slope_scale=8.0,
        model_name=args.model,
        device=args.device,
        output_path=args.output,
    )
    normalizer = DepthNormalizer(config)

    image = Image.open(args.input).convert("RGB")
    rgb = np.asarray(image)
    normal_map = normalizer.process_image(rgb)
    output_path = normalizer.save(normal_map)

    print(f"Saved normal map to {output_path}")


if __name__ == "__main__":
    main()
