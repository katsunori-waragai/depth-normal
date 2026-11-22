"""カラー画像から深度マップを推定し、法線マップおよび擬似シェーディング画像を生成するモジュール."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import warnings

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for real models
    torch = None


@dataclass
class NormalMapConfig:
    """法線マップ生成のパラメータおよび実行環境を保持する設定クラス.

    Attributes:
        slope_scale: 深度の勾配に対する傾斜スケール（法線の凹凸の強さ）。正の値である必要があります。
        model_name: 利用可能な場合に優先して使用する Depth-Anything-V2 のモデル名。
        device: 推論を実行するデバイス名（例: "cpu", "cuda"）。
    """

    slope_scale: float = 8.0
    model_name: Optional[str] = None
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

    インスタンス初期化時に推論モデルと前処理パイプラインをロードして保持し、
    複数画像の連続処理時に再ロードを発生させません。

    Attributes:
        config: 法線マップ生成に使用する設定オブジェクト。
    """

    config: NormalMapConfig = field(default_factory=NormalMapConfig)
    _model: Any = field(default=None, init=False, repr=False)
    _transform: Any = field(default=None, init=False, repr=False)
    _prepare_for_net: Any = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def __post_init__(self) -> None:
        """初期化後に推論モデルと前処理パイプラインをロードする.

        ライブラリが未インストールの場合やロードに失敗した場合は、
        内部状態を None のまま維持し、推論時にグレースケールフォールバックを使用します。
        """

        try:
            if torch is None:
                raise RuntimeError("PyTorch is not installed")

            device = torch.device(self.config.device)
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    warnings.warn("CUDA is not available. Falling back to CPU.", RuntimeWarning)
                    device = torch.device("cpu")
                elif device.index is not None and device.index >= torch.cuda.device_count():
                    warnings.warn(
                        f"CUDA device {device.index} is out of range. Falling back to CPU.",
                        RuntimeWarning,
                    )
                    device = torch.device("cpu")

            self._device = str(device)

            from depth_anything_v2.dpt import DepthAnythingV2
            from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet

            self._transform = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            self._prepare_for_net = PrepareForNet()
            model = DepthAnythingV2(device=self.config._device)
            if hasattr(model, "to"):
                model = model.to(self.config._device)
            if hasattr(model, "eval"):
                model.eval()
            self._model = model
        except Exception:
            self._model = None
            self._transform = None
            self._prepare_for_net = None

    def estimate_depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """カラー画像から深度マップを推定する.

        保持している Depth-Anything-V2 モデルを使用して推論を行います。
        モデルがロードされていない場合はグレースケールベースのフォールバックを使用します。

        Args:
            image_rgb: 入力となる RGB または RGBA の画像配列（形状: (H, W, 3) または (H, W, 4)）。

        Returns:
            0.0 〜 1.0 の範囲に正規化された 2 次元の深度マップ（dtype: float32）。

        Raises:
            ValueError: 入力画像が 3 次元配列でない場合、またはチャンネル数が 3・4 以外の場合。
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] not in (3, 4):
            raise ValueError(f"expected an RGB(A) image, got shape {image_rgb.shape}")

        if self._model is None:
            return self._estimate_depth_fallback(image_rgb)

        try:
            image = Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
            image = image.convert("RGB")
            image_tensor = self._prepare_for_net(image)
            image_tensor = self._transform(image_tensor)

            with torch.inference_mode():
                input_tensor = image_tensor.unsqueeze(0).to(self.config.device)
                depth = self._model(input_tensor)[0]
            depth = depth.detach().cpu().numpy()[0]
            depth = np.asarray(depth, dtype=np.float32)
            # Convert depth from the model representation into a normalized height map.
            depth = (depth - np.min(depth)) / max(np.ptp(depth), 1e-6)
            return depth
        except Exception:
            return self._estimate_depth_fallback(image_rgb)

    def _estimate_depth_fallback(self, image_rgb: np.ndarray) -> np.ndarray:
        """グレースケールの輝度値を用いて簡易的な深度マップを生成する（フォールバック用）.

        Args:
            image_rgb: 入力となる RGB または RGBA の画像配列。

        Returns:
            1e-3 〜 1.0 の範囲にクリップされた 2 次元の深度マップ（dtype: float32）。
        """
        color_code = (
            cv2.COLOR_RGB2GRAY if image_rgb.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
        )
        gray = cv2.cvtColor(image_rgb, color_code)
        depth = gray.astype(np.float32) / 255.0
        return np.clip(depth, 1e-3, 1.0)

    def compute_normals(self, depth: np.ndarray) -> np.ndarray:
        """深度マップから単位法線ベクトル（範囲: -1.0 〜 1.0）を計算する.

        Args:
            depth: 2 次元の深度マップ配列（高さ・幅ともに 2 ピクセル以上）。

        Returns:
            正規化された 3 次元単位法線ベクトル配列（形状: (H, W, 3), dtype: float32）。

        Raises:
            ValueError: depth が 2 次元配列でない場合、またはサイズが 2x2 未満の場合。
        """
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth must be a 2D array, got shape {depth.shape}")
        if depth.shape[0] < 2 or depth.shape[1] < 2:
            raise ValueError("depth must be at least 2x2")

        dz_dy, dz_dx = np.gradient(depth)
        dz_dy *= self.config.slope_scale
        dz_dx *= self.config.slope_scale

        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        return (normals / np.maximum(norms, 1e-6)).astype(np.float32)

    def normal_map_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """深度マップから色感度を強調した法線マップを生成する.

        Args:
            depth: 2 次元の深度マップ配列（高さ・幅ともに 2 ピクセル以上）。

        Returns:
            0.0 〜 1.0 の範囲に正規化された RGB 法線マップ配列（形状: (H, W, 3), dtype: float32）。
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
        """法線ベクトルと光源方向の内積に基づき、元画像の HSV 輝度(V)に擬似シェーディングを適用する.

        元の輝度 V に `1.0 + strength * cos(theta)` を掛けた後、
        V の最大値が 1.0 となるようにスケーリングして RGB に変換します。

        Args:
            image_rgb: 入力となる RGB 画像配列（dtype: uint8）。
            normals: compute_normals() で得られた単位法線ベクトル配列（範囲: -1.0 〜 1.0）。
            light_dir: 光源の方向ベクトル (x:右, y:下, z:手前)。内部で単位ベクトルに正規化されます。
            strength: シェーディングの強度係数。

        Returns:
            擬似シェーディング適用後の 0.0 〜 1.0 正規化 RGB 画像配列（形状: (H, W, 3), dtype: float32）。
        """
        light = np.asarray(light_dir, dtype=np.float32)
        light = light / max(float(np.linalg.norm(light)), 1e-6)

        # 法線ベクトルと光源方向ベクトルの内積（コサイン）
        cos_theta = np.sum(normals * light, axis=-1)

        # 1 + 係数 * cos()
        shading_factor = np.maximum(1.0 + strength * cos_theta, 0.0)

        # RGB(0.0~1.0) -> HSV(H: 0~360, S: 0.0~1.0, V: 0.0~1.0)
        rgb_float = image_rgb[:, :, :3].astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2HSV)

        # 輝度 V に係数を掛け、最大値が 1.0 になるように正規化
        v_shaded = hsv[:, :, 2] * shading_factor
        max_v = float(np.max(v_shaded))
        if max_v > 1e-6:
            v_shaded = v_shaded / max_v
        hsv[:, :, 2] = np.clip(v_shaded, 0.0, 1.0)

        # HSV -> RGB(0.0~1.0) に戻す
        shaded_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(shaded_rgb, 0.0, 1.0).astype(np.float32)

    def process_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """カラー画像から深度推定と法線マップ変換を一貫して実行する.

        Args:
            image_rgb: 入力となる RGB または RGBA の画像配列。

        Returns:
            0.0 〜 1.0 の範囲に正規化された RGB 法線マップ配列（dtype: float32）。
        """
        depth = self.estimate_depth_map(image_rgb)
        return self.normal_map_from_depth(depth)

    def save(self, image_float: np.ndarray, output_path: Path) -> Path:
        """0.0 〜 1.0 の RGB 画像配列を画像ファイルとして保存する.

        親ディレクトリが存在しない場合は自動的に作成します。

        Args:
            image_float: 0.0 〜 1.0 に正規化された RGB 画像配列。
            output_path: 保存先のファイルパス。

        Returns:
            保存されたファイルの Path オブジェクト。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            (image_float * 255.0).clip(0, 255).astype(np.uint8)
        ).save(path)
        return path

    def run_file(
            self,
            input_path: Path,
            output_path: Path,
            input_save_path: Optional[Path] = None,
            shaded_save_path: Optional[Path] = None,
            light_dir: tuple[float, float, float] = (-1.0, -1.0, 1.0),
            strength: float = 5.0,
        ) -> tuple[Path, Path, Path]:
            """画像ファイルを読み込み、元画像コピー・法線マップ・擬似シェーディング画像を保存する.

            Args:
                input_path: 入力画像のファイルパス。
                output_path: 出力する法線マップ画像のファイルパス。
                input_save_path: 元画像の保存先パス。省略時は output_path の末尾に "_input" を付与します。
                shaded_save_path: 擬似シェーディング画像の保存先パス。省略時は "_shaded" を付与します。
                light_dir: 擬似シェーディングに使用する光源の方向ベクトル。
                strength: 擬似シェーディングの強度係数。

            Returns:
                (保存された元画像の Path, 保存された法線マップの Path, 保存されたシェーディング画像の Path) のタプル。
            """
            in_path = Path(input_path)
            out_path = Path(output_path)

            in_save_path = (
                Path(input_save_path)
                if input_save_path is not None
                else out_path.with_name(f"{out_path.stem}_input{out_path.suffix}")
            )
            shade_save_path = (
                Path(shaded_save_path)
                if shaded_save_path is not None
                else out_path.with_name(f"{out_path.stem}_shaded{out_path.suffix}")
            )

            # 1. 入力画像の読み込みと保存
            image = Image.open(in_path).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
            in_save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(in_save_path)

            # 2. 深度推定と単位法線ベクトルの計算
            depth = self.estimate_depth_map(rgb)
            normals = self.compute_normals(depth)

            # 3. 法線マップの生成と保存
            normal_map = np.clip((normals + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
            saved_out_path = self.save(normal_map, out_path)

            # 4. 擬似シェーディング画像の生成と保存
            light_dirs: list[tuple[float, float, float]] = [
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ]

            saved_shade_paths = []
            for i, light_dir in enumerate(light_dirs):
                shaded_rgb = self.apply_pseudo_shading(
                rgb, normals, light_dir=light_dir, strength=strength
                )
                shade_save_path = (
                Path(shaded_save_path)
                if shaded_save_path is not None
                else out_path.with_name(f"{out_path.stem}_shaded_{i}{out_path.suffix}")
                )

                saved_shade_path = self.save(shaded_rgb, shade_save_path)
                saved_shade_paths.append(saved_shade_path)

            return in_save_path, saved_out_path, saved_shade_paths

    def run_demo(
        self,
        output_path: Path,
        input_save_path: Optional[Path] = None,
        shaded_save_path: Optional[Path] = None,
        light_dir: tuple[float, float, float] = (-1.0, -1.0, 1.0),
        strength: float = 5.0,
        ) -> tuple[Path, Path, Path]:
        """デフォルトのサンプル画像（figures/coffee.png）を使用して標準パイプラインを実行する."""
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
    """コマンドライン引数を解析し、法線マップ生成処理を実行するエントリーポイント."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Estimate a depth map and visualize it as a normal-map image."
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
        help="Optional DepthAnything model name to prefer when available.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run the model on, e.g. cpu or cuda."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo using coffee image and save input, normal map, and shaded images.",
    )
    args = parser.parse_args()

    if not args.demo and args.input is None:
        parser.error(
            "the following arguments are required: input (unless --demo is specified)"
        )

    config = NormalMapConfig(
        slope_scale=8.0,
        model_name=args.model,
        device=args.device,
    )
    normalizer = DepthNormalizer(config)

    if args.demo:
        input_saved, output_saved, shaded_saved = normalizer.run_demo(args.output)
        print(f"Saved demo input image to {input_saved}")
        print(f"Saved demo normal map to {output_saved}")
        print(f"Saved demo shaded image to {shaded_saved}")
    else:
        output_path = normalizer.run_file(args.input, args.output)
        print(f"Saved normal map to {output_path}")


if __name__ == "__main__":
    main()
