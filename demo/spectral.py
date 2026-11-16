"""Native band metadata, visualization and full-resolution exports."""

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from scipy.io import savemat

from .inference import MODEL_REPO, MODEL_REVISION, SOURCE_REVISION

# visualization/show_simulation.m: lam31, with 1-based channel numbers.
WAVELENGTHS = np.arange(400, 701, 10, dtype=np.float32)
DEFAULT_WAVELENGTH_TEXT = ", ".join(f"{x:g}" for x in WAVELENGTHS)


def parse_channels(text):
    if not text or text.strip().lower() in ("all", "全部"):
        return list(range(31))
    result = []
    for token in re.split(r"[,，;\s]+", text.strip()):
        if not token:
            continue
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            raise ValueError("Use channel numbers 1–31, for example 1, 9, 13, 19, 27 or 1-31.")
        parts = [int(x) for x in token.split("-")]
        start, end = parts[0], parts[-1]
        if not 1 <= start <= end <= 31:
            raise ValueError("Channel numbers must be within 1–31, with increasing ranges.")
        result.extend(range(start - 1, end))
    if not result:
        raise ValueError("Select at least one channel.")
    return list(dict.fromkeys(result))


def parse_wavelengths(text):
    try:
        waves = np.array([float(x) for x in re.split(r"[,，;\s]+", text.strip()) if x])
    except (ValueError, AttributeError) as exc:
        raise ValueError("Enter 31 numeric wavelength labels, separated by commas.") from exc
    if waves.shape != (31,) or not np.isfinite(waves).all():
        raise ValueError("Exactly 31 finite wavelength labels are required.")
    if waves[0] < 380 or waves[-1] > 780 or np.any(np.diff(waves) <= 0):
        raise ValueError("Wavelength labels must increase strictly within 380–780 nm.")
    return waves


def wavelength_color(wavelength):
    """Approximate visible hue for illustration, not calibrated colorimetry."""
    w = float(wavelength)
    if w < 440:
        rgb = [(440 - w) / 60, 0, 1]
    elif w < 490:
        rgb = [0, (w - 440) / 50, 1]
    elif w < 510:
        rgb = [0, 1, (510 - w) / 20]
    elif w < 580:
        rgb = [(w - 510) / 70, 1, 0]
    elif w < 645:
        rgb = [1, (645 - w) / 65, 0]
    else:
        rgb = [1, 0, 0]
    return np.clip(rgb, 0, 1).astype(np.float32)


def display_band(cube, index, waves=WAVELENGTHS, brightness=1.0, color=True):
    if not 0 <= int(index) < 31:
        raise ValueError("Choose a channel from 1–31.")
    if not 0.1 <= float(brightness) <= 3:
        raise ValueError("Display brightness must be in 0.1–3.0.")
    intensity = np.clip(cube[..., int(index)] * float(brightness), 0, 1)
    hue = wavelength_color(waves[int(index)]) if color else np.ones(3)
    return np.rint(intensity[..., None] * hue * 255).astype(np.uint8)


def gallery(cube, channels, waves, brightness, color, thumbnail_size=None):
    items = []
    for index in channels:
        preview = display_band(cube, index, waves, brightness, color)
        if thumbnail_size:
            image = Image.fromarray(preview)
            image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            preview = np.asarray(image)
        items.append((preview, f"Ch {index + 1:02d} · {waves[index]:g} nm"))
    return items


def spectrum_figure(cube, waves=WAVELENGTHS, xy=None):
    if xy is None:
        values, suffix = cube.mean(axis=(0, 1)), "spatial mean"
    else:
        x, y = xy
        values, suffix = cube[y, x], f"pixel ({x}, {y})"
    figure = Figure(figsize=(6.4, 2.2), layout="constrained")
    ax = figure.subplots()
    ax.plot(waves, values, color="#6366f1", linewidth=1.8)
    ax.set(xlabel="Wavelength (nm)", ylabel="Reconstructed intensity", title=f"Spectrum · {suffix}")
    ax.grid(alpha=0.2)
    return figure


def metadata(result):
    return dict(
        model=result["model"],
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        source_revision=SOURCE_REVISION,
        input="RGB image",
        output="31-band HSI",
        native_wavelengths_nm=WAVELENGTHS.tolist(),
        cube_shape=list(result["cube"].shape),
        original_width_height=list(result["original_size"]),
        output_width_height=list(result["output_size"]),
        preprocessing="RGB global min-max normalization; uniform images use RGB/255",
        processing=result["processing"],
        self_ensemble=result["ensemble"],
    )


def export_cube(result):
    path = Path(result["output_dir"]) / "reconstruction.mat"
    savemat(
        path,
        dict(
            cube=result["cube"],
            raw_cube=result["raw_cube"],
            wavelengths_nm=WAVELENGTHS,
            metadata_json=json.dumps(metadata(result)),
        ),
        do_compression=True,
    )
    return str(path)


def export_channels(result, channels, waves, brightness, color):
    path = Path(result["output_dir"]) / "spectral_channels.zip"
    meta = metadata(result)
    meta.update(
        display_channels_1based=[i + 1 for i in channels],
        display_wavelengths_nm=np.asarray(waves).tolist(),
        brightness=float(brightness),
        pseudocolor=bool(color),
        note="Edited wavelengths are display labels; the trained 31-band output grid is unchanged.",
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(meta, indent=2))
        for index in channels:
            for folder, use_color in [("preview", color), ("grayscale", False)]:
                buffer = io.BytesIO()
                Image.fromarray(
                    display_band(result["cube"], index, waves, brightness, use_color)
                ).save(buffer, format="PNG")
                archive.writestr(
                    f"{folder}/channel_{index + 1:02d}_{waves[index]:g}nm.png", buffer.getvalue()
                )
    return str(path)
