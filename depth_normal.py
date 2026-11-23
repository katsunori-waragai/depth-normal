"""カラー画像から深度マップを推定し、法線マップおよび擬似シェーディング画像を生成するモジュール.

深度推定には Hugging Face Transformers の depth-estimation pipeline を使用します。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import warnings

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency for real models
    torch = None
    pipeline = None


@dataclass
class NormalMapConfig:
    """法線マップ生成のパラメータおよび実行環境を保持する設定クラス.

    Attributes:
        slope_scale: 深度の勾配に対する傾斜スケール（法線の凹凸の強さ）。
            正の値である必要があります。
        model_name: Hugging Face Hub の深度推定モデル ID。
        device: 推論を実行するデバイス名（例: "cpu", "cuda", "cuda:0"）。
    """

    slope_scale: float = 8.0
    model_name: Optional[str] = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "cpu"

    def __post_init__(self) -> None:
        """初期化後にフィールドの型変換とバリデーションを実行する.

        Raises:
            ValueError: slope_scale が 0 以下の場合。
        """
        self.slope_scale = float(self.slope_scale)
        if self.slope_scale <= 0:
            raise ValueError("slope_scale must be positive")
        self.device = str(self.device)


@dataclass
class DepthNormalizer:
    """カラー画像から深度推定、法線マップ変換、擬似シェーディングを行うクラス.

    インスタンス初期化時に推論モデルをロードして保持し、
    複数画像の連続処理時にモデルの再ロードを発生させません。

    Attributes:
        config: 法線マップ生成に使用する設定オブジェクト。
    """

    config: NormalMapConfig = field(default_factory=NormalMapConfig)
    _model: Any = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def __post_init__(self) -> None:
        """Hugging Face Transformers の深度推定 pipeline をロードする.

        ライブラリが未インストールの場合やロードに失敗した場合は、
        内部状態を None のまま維持し、推論時にグレースケール
        フォールバックを使用します。
        """
        try:
            if torch is None or pipeline is None:
                raise RuntimeError(
                    "PyTorch or Transformers is not installed"
                )

            device = torch.device(self.config.device)

            if device.type == "cuda":
                if not torch.cuda.is_available():
                    warnings.warn(
                        "CUDA is not available. Falling back to CPU.",
                        RuntimeWarning,
                    )
                    device = torch.device("cpu")
                elif (
                    device.index is not None
                    and device.index >= torch.cuda.device_count()
                ):
                    warnings.warn(
                        f"CUDA device {device.index} is out of range. "
                        "Falling back to CPU.",
                        RuntimeWarning,
                    )
                    device = torch.device("cpu")

            self._device = str(device)

            model_name = (
                self.config.model_name
                or "depth-anything/Depth-Anything-V2-Small-hf"
            )

            # Transformers pipeline の device は CPU=-1、CUDA=n と指定する。
            device_id = (
                device.index if device.index is not None else 0
            ) if device.type == "cuda" else -1

            self._model = pipeline(
                task="depth-estimation",
                model=model_name,
                device=device_id,
            )

        except Exception as exc:
            warnings.warn(
                f"Failed to initialize Hugging Face depth model "
                f"({type(exc).__name__}: {exc}). "
                "Falling back to grayscale depth estimation.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = None

    def estimate_depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """カラー画像から深度マップを推定する.

        保持している Hugging Face 深度推定モデルを使用して推論します。
        モデルがロードされていない場合はグレースケールベースの
        フォールバックを使用します。

        Args:
            image_rgb: 入力となる RGB または RGBA の画像配列
                （形状: (H, W, 3) または (H, W, 4)）。

        Returns:
            0.0〜1.0 の範囲に正規化された 2 次元の深度マップ
            （dtype: float32）。

        Raises:
            ValueError: 入力画像が 3 次元配列でない場合、
                またはチャンネル数が 3・4 以外の場合。
        """
        image_rgb = np.asarray(image_rgb)

        if image_rgb.ndim != 3 or image_rgb.shape[2] not in (3, 4):
            raise ValueError(
                f"expected an RGB(A) image, got shape {image_rgb.shape}"
            )

        if image_rgb.dtype != np.uint8:
            raise ValueError(
                f"image_rgb must have dtype uint8, got {image_rgb.dtype}"
            )

        if self._model is None:
            return self._estimate_depth_fallback(image_rgb)

        try:
            # 引数は RGB 配列として扱う。BGR/RGB の色変換は行わない。
            image = Image.fromarray(image_rgb[:, :, :3], mode="RGB")

            result = self._model(image)

            # predicted_depth は表示用画像ではなくモデルの Tensor 出力。
            depth_tensor = result["predicted_depth"]
            depth = depth_tensor.detach().cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)

            # pipeline の通常出力形状 (1, H, W) から (H, W) にする。
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            elif depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]

            if depth.ndim != 2:
                raise ValueError(
                    "unexpected predicted_depth shape: "
                    f"{depth.shape}"
                )

            # 現行実装との互換性を優先し、画像ごとに min-max 正規化する。
            depth_min = float(np.min(depth))
            depth_range = float(np.ptp(depth))
            depth = (depth - depth_min) / max(depth_range, 1e-6)

            return depth.astype(np.float32)

        except Exception as exc:
            warnings.warn(
                f"Depth estimation inference failed "
                f"({type(exc).__name__}: {exc}). "
                "Falling back to grayscale depth estimation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._estimate_depth_fallback(image_rgb)

    def _estimate_depth_fallback(self, image_rgb: np.ndarray) -> np.ndarray:
        """グレースケールの輝度値を用いて簡易的な深度マップを生成する.

        Args:
            image_rgb: 入力となる RGB または RGBA の uint8 画像配列。

        Returns:
            1e-3〜1.0 の範囲にクリップされた 2 次元の深度マップ
            （dtype: float32）。
        """
        color_code = (
            cv2.COLOR_RGB2GRAY
            if image_rgb.shape[2] == 3
            else cv2.COLOR_RGBA2GRAY
        )
        gray = cv2.cvtColor(image_rgb, color_code)
        depth = gray.astype(np.float32) / 255.0
        return np.clip(depth, 1e-3, 1.0)

    def compute_normals(self, depth: np.ndarray) -> np.ndarray:
        """深度マップから単位法線ベクトル（範囲: -1.0〜1.0）を計算する.

        Args:
            depth: 2 次元の深度マップ配列（高さ・幅ともに 2 ピクセル以上）。

        Returns:
            正規化された 3 次元単位法線ベクトル配列
            （形状: (H, W, 3), dtype: float32）。

        Raises:
            ValueError: depth が 2 次元配列でない場合、
                またはサイズが 2x2 未満の場合。
        """
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(
                f"depth must be a 2D array, got shape {depth.shape}"
            )
        if depth.shape[0] < 2 or depth.shape[1] < 2:
            raise ValueError("depth must be at least 2x2")

        dz_dy, dz_dx = np.gradient(depth)
        dz_dy *= self.config.slope_scale
        dz_dx *= self.config.slope_scale

        normals = np.stack(
            [-dz_dx, -dz_dy, np.ones_like(depth)],
            axis=-1,
        )
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        return (normals / np.maximum(norms, 1e-6)).astype(np.float32)

    def normal_map_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """深度マップから RGB 法線マップを生成する.

        Args:
            depth: 2 次元の深度マップ配列（高さ・幅ともに 2 ピクセル以上）。

        Returns:
            0.0〜1.0 に正規化された RGB 法線マップ配列
            （形状: (H, W, 3), dtype: float32）。
        """
        normals = self.compute_normals(depth)
        normal_map = (normals + 1.0) * 0.5
        return np.clip(normal_map, 0.0, 1.0).astype(np.float32)

    def apply_pseudo_shading(
        self,
        image_rgb: np.ndarray,
        normals: np.ndarray,
        light_dir: tuple[float, float, float] = (-1.0, -1.0, 1.0),
        strength: float = 0.6,
    ) -> np.ndarray:
        """法線ベクトルと光源方向の内積に基づき、元画像に擬似シェーディングを適用する.

        Args:
            image_rgb: 入力となる RGB 画像配列（dtype: uint8）。
            normals: compute_normals() で得られた単位法線ベクトル配列。
            light_dir: 光源の方向ベクトル (x:右, y:下, z:手前)。
            strength: シェーディングの強度係数。

        Returns:
            擬似シェーディング適用後の RGB 画像配列
            （範囲: 0.0〜1.0、dtype: float32）。
        """
        light = np.asarray(light_dir, dtype=np.float32)
        light_norm = float(np.linalg.norm(light))
        if light_norm <= 1e-6:
            raise ValueError("light_dir must be a non-zero vector")
        light = light / light_norm

        cos_theta = np.sum(normals * light, axis=-1)
        shading_factor = np.maximum(1.0 + strength * cos_theta, 0.0)

        rgb_float = image_rgb[:, :, :3].astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2HSV)

        v_shaded = hsv[:, :, 2] * shading_factor
        max_v = float(np.max(v_shaded))
        if max_v > 1e-6:
            v_shaded = v_shaded / max_v
        hsv[:, :, 2] = np.clip(v_shaded, 0.0, 1.0)

        shaded_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(shaded_rgb, 0.0, 1.0).astype(np.float32)

    def process_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """カラー画像から深度推定と法線マップ変換を一貫して実行する.

        Args:
            image_rgb: 入力となる RGB または RGBA の uint8 画像配列。

        Returns:
            0.0〜1.0 に正規化された RGB 法線マップ配列（dtype: float32）。
        """
        depth = self.estimate_depth_map(image_rgb)
        return self.normal_map_from_depth(depth)

    def save(self, image_float: np.ndarray, output_path: Path) -> Path:
        """0.0〜1.0 の RGB 画像配列を画像ファイルとして保存する.

        親ディレクトリが存在しない場合は自動的に作成します。

        Args:
            image_float: 0.0〜1.0 に正規化された RGB 画像配列。
            output_path: 保存先のファイルパス。

        Returns:
            保存されたファイルの Path オブジェクト。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image_uint8 = (
            np.asarray(image_float) * 255.0
        ).clip(0, 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(path)
        return path

    def run_file(
        self,
        input_path: Path,
        output_path: Path,
        input_save_path: Optional[Path] = None,
        shaded_save_path: Optional[Path] = None,
        light_dir: tuple[float, float, float] = (-1.0, -1.0, 1.0),
        strength: float = 5.0,
    ) -> tuple[Path, Path, list[Path]]:
        """入力画像・法線マップ・4方向の擬似シェーディング画像を保存する.

        Args:
            input_path: 入力画像のファイルパス。
            output_path: 出力する法線マップ画像のファイルパス。
            input_save_path: 元画像の保存先。省略時は "_input" を付与する。
            shaded_save_path: シェーディング画像の保存先の基準パス。
                省略時は方向ごとに "_shaded_0" などを付与する。
                指定時も4方向を別ファイルに保存するため、末尾に番号を付与する。
            light_dir: 互換性のため残している引数。現在は4方向の光源を使用する。
            strength: シェーディングの強度係数。

        Returns:
            (元画像の Path, 法線マップの Path, シェーディング画像の Path のリスト)。
        """
        in_path = Path(input_path)
        out_path = Path(output_path)

        in_save_path = (
            Path(input_save_path)
            if input_save_path is not None
            else out_path.with_name(f"{out_path.stem}_input{out_path.suffix}")
        )

        image = Image.open(in_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        in_save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(in_save_path)

        depth = self.estimate_depth_map(rgb)
        normals = self.compute_normals(depth)

        normal_map = np.clip(
            (normals + 1.0) * 0.5, 0.0, 1.0
        ).astype(np.float32)
        saved_out_path = self.save(normal_map, out_path)

        light_dirs: list[tuple[float, float, float]] = [
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ]

        saved_shade_paths: list[Path] = []
        for i, current_light_dir in enumerate(light_dirs):
            shaded_rgb = self.apply_pseudo_shading(
                rgb,
                normals,
                light_dir=current_light_dir,
                strength=strength,
            )

            if shaded_save_path is None:
                shade_path = out_path.with_name(
                    f"{out_path.stem}_shaded_{i}{out_path.suffix}"
                )
            else:
                base_path = Path(shaded_save_path)
                shade_path = base_path.with_name(
                    f"{base_path.stem}_{i}{base_path.suffix}"
                )

            saved_shade_paths.append(self.save(shaded_rgb, shade_path))

        return in_save_path, saved_out_path, saved_shade_paths

    def run_demo(
        self,
        output_path: Path,
        input_save_path: Optional[Path] = None,
        shaded_save_path: Optional[Path] = None,
        light_dir: tuple[float, float, float] = (-1.0, -1.0, 1.0),
        strength: float = 5.0,
    ) -> tuple[Path, Path, list[Path]]:
        """デフォルトのサンプル画像（figures/coffee.png）で処理を実行する."""
        demo_input = Path("figures/coffee.png")
        return self.run_file(
            input_path=demo_input,
            output_path=output_path,
            input_save_path=input_save_path,
            shaded_save_path=shaded_save_path,
            light_dir=light_dir,
            strength=strength,
        )


def main() -> None:
    """コマンドライン引数を解析し、法線マップ生成処理を実行する."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Estimate a depth map and visualize it as a normal-map image."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input image path (optional when --demo is used).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("normal_map.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Hugging Face model ID. "
            "Default: depth-anything/Depth-Anything-V2-Small-hf"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the model on, e.g. cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run demo using figures/coffee.png and save input, "
            "normal map, and shaded images."
        ),
    )
    args = parser.parse_args()

    if not args.demo and args.input is None:
        parser.error(
            "the following arguments are required: input "
            "(unless --demo is specified)"
        )

    config = NormalMapConfig(
        slope_scale=8.0,
        model_name=args.model,
        device=args.device,
    )
    normalizer = DepthNormalizer(config)

    if args.demo:
        input_saved, output_saved, shaded_saved = normalizer.run_demo(
            args.output
        )
        print(f"Saved demo input image to {input_saved}")
        print(f"Saved demo normal map to {output_saved}")
        for path in shaded_saved:
            print(f"Saved demo shaded image to {path}")
    else:
        input_saved, output_saved, shaded_saved = normalizer.run_file(
            args.input,
            args.output,
        )
        print(f"Saved input image to {input_saved}")
        print(f"Saved normal map to {output_saved}")
        for path in shaded_saved:
            print(f"Saved shaded image to {path}")


if __name__ == "__main__":
    main()
