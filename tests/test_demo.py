"""Contracts for faithful RGB preprocessing, spatial reconstruction and exports."""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import loadmat
from torch import nn

from demo.inference import MODELS, Reconstructor, forward_ensemble, prepare_rgb
from demo.spectral import (
    DEFAULT_WAVELENGTH_TEXT,
    WAVELENGTHS,
    display_band,
    export_channels,
    export_cube,
    parse_channels,
    parse_wavelengths,
)


def test_native_wavelengths_match_upstream():
    np.testing.assert_array_equal(WAVELENGTHS, np.arange(400, 701, 10))
    assert len(MODELS) == 11


def test_rgb_order_and_official_global_minmax_normalization():
    array = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    tensor, before, after = prepare_rgb(Image.fromarray(array))
    np.testing.assert_allclose(tensor[0].permute(1, 2, 0), (array.astype(np.float32) - 10) / 50)
    assert before == after == (2, 1)


@pytest.mark.parametrize("value", [0, 100, 255])
def test_constant_images_are_finite(value):
    tensor, _, _ = prepare_rgb(Image.new("RGB", (5, 7), (value,) * 3))
    assert torch.isfinite(tensor).all()
    torch.testing.assert_close(tensor, torch.full_like(tensor, value / 255))


def test_max_size_keeps_aspect_ratio_and_never_upscales():
    _, before, after = prepare_rgb(Image.new("RGB", (960, 480)), max_size=256)
    assert before == (960, 480) and after == (256, 128)
    _, _, after = prepare_rgb(Image.new("RGB", (20, 10)), max_size=256)
    assert after == (20, 10)


class PointwiseSpectralModel(nn.Module):
    def forward(self, x):
        weights = x.new_tensor([0.2, 0.3, 0.5])[None, :, None, None]
        intensity = (x * weights).sum(1, keepdim=True)
        return intensity.expand(-1, 31, -1, -1)


@pytest.mark.parametrize("ensemble", [False, True])
def test_padding_crop_and_eight_way_transform_inversion(ensemble):
    tensor = torch.rand(1, 3, 5, 13)
    expected = PointwiseSpectralModel()(tensor)
    output = forward_ensemble(PointwiseSpectralModel(), tensor, ensemble=ensemble)
    assert output.shape == (1, 31, 5, 13)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("model_key", ["mst_plus_plus", "awan"])
def test_large_inputs_are_processed_in_one_full_image_call(monkeypatch, model_key):
    seen = []

    class RecordingModel(PointwiseSpectralModel):
        def forward(self, tensor):
            seen.append(tuple(tensor.shape))
            return super().forward(tensor)

    engine = Reconstructor(device="cpu")
    monkeypatch.setattr(engine, "load_model", lambda key: RecordingModel())
    result = engine.reconstruct(
        Image.new("RGB", (513, 289), (40, 100, 200)), key=model_key, max_size=0
    )
    assert len(seen) == 1
    assert seen[0][-2] >= 289 and seen[0][-1] >= 513
    assert result["cube"].shape == (289, 513, 31)
    assert result["processing"] == "Full image"


def test_channel_ranges_and_label_validation():
    assert parse_channels("1, 9, 13, 19, 27") == [0, 8, 12, 18, 26]
    assert parse_channels("8-12, 8，31") == [7, 8, 9, 10, 11, 30]
    assert parse_channels("all") == list(range(31))
    np.testing.assert_array_equal(parse_wavelengths(DEFAULT_WAVELENGTH_TEXT), WAVELENGTHS)
    for invalid in ["32", "0", "10-3", "a"]:
        with pytest.raises(ValueError):
            parse_channels(invalid)
    for invalid in ["400,500", "nan," * 31, ",".join(["500"] * 31)]:
        with pytest.raises(ValueError):
            parse_wavelengths(invalid)


def test_preview_does_not_modify_hsi():
    cube = np.full((3, 4, 31), 0.25, dtype=np.float32)
    before = cube.copy()
    preview = display_band(cube, 8, brightness=2, color=False)
    np.testing.assert_array_equal(preview, np.full((3, 4, 3), 128, dtype=np.uint8))
    np.testing.assert_array_equal(cube, before)


def test_native_cube_and_custom_label_exports_are_separate(tmp_path):
    raw = np.linspace(-0.2, 1.2, 3 * 4 * 31, dtype=np.float32).reshape(3, 4, 31)
    cube = raw.clip(0, 1)
    result = dict(
        cube=cube,
        raw_cube=raw,
        model="mst_plus_plus",
        output_dir=str(tmp_path),
        original_size=(4, 3),
        output_size=(4, 3),
        processing="Full image",
        ensemble=False,
    )
    mat = loadmat(export_cube(result))
    np.testing.assert_array_equal(mat["cube"], cube)
    np.testing.assert_array_equal(mat["raw_cube"], raw)
    np.testing.assert_array_equal(mat["wavelengths_nm"][0], WAVELENGTHS)
    labels = WAVELENGTHS.copy()
    labels[0] = 405
    archive = export_channels(result, [0, 30], labels, brightness=2, color=True)
    with zipfile.ZipFile(archive) as z:
        metadata = json.loads(z.read("metadata.json"))
        assert metadata["display_wavelengths_nm"][0] == 405
        assert metadata["native_wavelengths_nm"][0] == 400
        assert len([p for p in z.namelist() if p.endswith(".png")]) == 4
    np.testing.assert_array_equal(result["cube"], cube)
