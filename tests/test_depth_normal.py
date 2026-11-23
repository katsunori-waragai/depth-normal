import numpy as np

from depth_normal import DepthNormalizer, NormalMapConfig


def _make_normalizer(slope_scale: float = 8.0) -> DepthNormalizer:
    return DepthNormalizer(NormalMapConfig(slope_scale=slope_scale, model_name=None, device="cpu"))


def test_depth_to_normal_map_returns_rgb_image():
    depth = np.linspace(1.0, 2.0, 16, dtype=np.float32).reshape(4, 4)
    normal_map = _make_normalizer().normal_map_from_depth(depth)

    assert normal_map.shape == (4, 4, 3)
    assert normal_map.dtype == np.float32
    assert np.all(np.isfinite(normal_map))
    assert np.min(normal_map) >= 0.0
    assert np.max(normal_map) <= 1.0


def test_depth_to_normal_map_handles_flat_depth():
    depth = np.ones((8, 8), dtype=np.float32)
    normal_map = _make_normalizer().normal_map_from_depth(depth)

    assert normal_map.shape == (8, 8, 3)
    assert np.allclose(normal_map[:, :, 0], 0.5, atol=1e-6)
    assert np.allclose(normal_map[:, :, 1], 0.5, atol=1e-6)
    assert np.allclose(normal_map[:, :, 2], 1.0, atol=1e-6)


def test_depth_to_normal_map_has_strong_color_response_to_slopes():
    x = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    depth = xx**2 + yy**2
    normal_map = _make_normalizer().normal_map_from_depth(depth)

    assert np.ptp(normal_map[:, :, 0]) > 0.3
    assert np.ptp(normal_map[:, :, 1]) > 0.3
    assert np.ptp(normal_map[:, :, 2]) > 0.1


def test_dataclass_pipeline_works():
    config = NormalMapConfig(slope_scale=4.0, model_name=None, device="cpu")
    normalizer = DepthNormalizer(config)
    depth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    normal_map = normalizer.normal_map_from_depth(depth)

    assert normal_map.shape == (4, 4, 3)
    assert np.all(np.isfinite(normal_map))
