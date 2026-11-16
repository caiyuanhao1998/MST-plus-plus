"""Use the upstream RGB networks and official pretrained weights directly."""

import gc
import importlib.util
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "CaiYuanhao/MST_pp"
MODEL_REVISION = "5aee97d596993c0c955045fced6231789c04c8a0"
SOURCE_REVISION = "23261eaa76b2a3766339a783820b4e588e016566"
# Matches test_develop_code/architecture/__init__.py, including AWAN.
MODELS = {
    "mst_plus_plus": ("MST++", "MST_Plus_Plus", "MST_Plus_Plus", {}),
    "mst": ("MST-L", "MST", "MST", dict(dim=31, stage=2, num_blocks=[4, 7, 5])),
    "hdnet": ("HDNet", "HDNet", "HDNet", {}),
    "restormer": ("Restormer", "Restormer", "Restormer", {}),
    "mprnet": ("MPRNet", "MPRNet", "MPRNet", dict(num_cab=4)),
    "hinet": ("HINet", "hinet", "HINet", dict(depth=4)),
    "mirnet": ("MIRNet", "MIRNet", "MIRNet", dict(n_RRG=3, n_MSRB=1, height=3, width=1)),
    "awan": ("AWAN", "AWAN", "AWAN", {}),
    "hrnet": ("HRNet", "hrnet", "SGN", {}),
    "hscnn_plus": ("HSCNN+", "HSCNN_Plus", "HSCNN_Plus", {}),
    "edsr": ("EDSR", "edsr", "EDSR", {}),
}


@lru_cache(maxsize=None)
def architecture(filename):
    path = ROOT / "test_develop_code/architecture" / f"{filename}.py"
    spec = importlib.util.spec_from_file_location(f"mstpp_demo_{filename}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(key):
    if key not in MODELS:
        raise ValueError("Choose a reconstruction model from the dropdown.")
    _, module, constructor, kwargs = MODELS[key]
    return getattr(architecture(module), constructor)(**kwargs)


def prepare_rgb(image, max_size=512):
    if image is None:
        raise ValueError("Upload an RGB image first.")
    if int(max_size) not in (0, 256, 512, 1024, 2048):
        raise ValueError("Choose a supported maximum image size.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    if int(max_size):
        image.thumbnail((int(max_size), int(max_size)), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32)
    low, high = float(rgb.min()), float(rgb.max())
    # Faithful predict_code/test.py min/max preprocessing; guard uniform inputs.
    rgb = (rgb - low) / (high - low) if high > low else rgb / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor, original_size, image.size


def forward_once(model, tensor, alignment=8):
    h, w = tensor.shape[-2:]
    ph = max(32, ((h + alignment - 1) // alignment) * alignment) - h
    pw = max(32, ((w + alignment - 1) // alignment) * alignment) - w
    mode = "reflect" if ph < h and pw < w else "replicate"
    result = model(F.pad(tensor, (0, pw, 0, ph), mode=mode))
    if isinstance(result, (list, tuple)):
        result = result[0]
    if result.shape[:2] != (tensor.shape[0], 31):
        raise RuntimeError(f"Unexpected model output shape: {tuple(result.shape)}")
    return result[..., :h, :w]


def forward_ensemble(model, tensor, alignment=8, ensemble=False):
    if not ensemble:
        return forward_once(model, tensor, alignment)
    output = None
    # The same eight flips/transpositions used by predict_code/test.py.
    for xflip in (False, True):
        for yflip in (False, True):
            for transpose in (False, True):
                sample = tensor
                if xflip:
                    sample = sample.flip(3)
                if yflip:
                    sample = sample.flip(2)
                if transpose:
                    sample = sample.transpose(2, 3)
                result = forward_once(model, sample, alignment)
                if transpose:
                    result = result.transpose(2, 3)
                if yflip:
                    result = result.flip(2)
                if xflip:
                    result = result.flip(3)
                output = result.clone() if output is None else output + result
    return output / 8


class Reconstructor:
    def __init__(self, weights_dir=None, device="auto"):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use auto, cpu, cuda, or cuda:N for --device.")
        if self.device.type == "cuda" and (
            not torch.cuda.is_available() or (self.device.index or 0) >= torch.cuda.device_count()
        ):
            raise ValueError(f"CUDA device is unavailable: {self.device}")
        self.weights_dirs = (
            [Path(weights_dir).expanduser().resolve()]
            if weights_dir is not None
            else [
                ROOT / "test_develop_code/model_zoo",
                ROOT / "test_challenge_code/model_zoo",
                ROOT / "pretrained_weights",
            ]
        )
        self.lock = Lock()
        self.model = None
        self.loaded_key = None
        self.loaded_stamp = None

    def _checkpoint_path(self, key):
        for directory in self.weights_dirs:
            path = directory / f"{key}.pth"
            if path.is_file():
                return path
        return self.weights_dirs[0] / f"{key}.pth"

    def available_models(self):
        return [
            (spec[0], key) for key, spec in MODELS.items() if self._checkpoint_path(key).is_file()
        ]

    def load_model(self, key):
        if key not in MODELS:
            raise ValueError("Choose a reconstruction model from the dropdown.")
        path = self._checkpoint_path(key)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {key}.pth. Prepare the model zoo following README "
                "section 3, or provide --weights-dir for your existing checkpoints."
            )
        stamp = (path, path.stat().st_mtime_ns, path.stat().st_size)
        if key == self.loaded_key and stamp == self.loaded_stamp:
            return self.model
        self.model, self.loaded_key = None, None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        model = build_model(key)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        self.model = model.eval().to(self.device)
        self.loaded_key, self.loaded_stamp = key, stamp
        return self.model

    @torch.inference_mode()
    def reconstruct(
        self,
        image,
        key="mst_plus_plus",
        max_size=512,
        ensemble=False,
        progress=None,
    ):
        report = progress or (lambda *args, **kwargs: None)
        tensor, original_size, output_size = prepare_rgb(image, max_size)
        with self.lock:
            started = perf_counter()
            report(0.05, desc="Loading reconstruction model")
            model = self.load_model(key)
            tensor = tensor.to(self.device)
            report(0.2, desc=f"Reconstructing 31 bands with {MODELS[key][0]}")
            try:
                prediction = forward_ensemble(
                    model, tensor, 16 if key == "hinet" else 8, bool(ensemble)
                )
                raw_cube = prediction[0].permute(1, 2, 0).cpu().numpy().copy()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise ValueError(
                    "GPU memory is full for full-image inference. Reduce Maximum image size (try 256 px for AWAN)."
                ) from exc
            if raw_cube.shape != (output_size[1], output_size[0], 31):
                raise RuntimeError(f"Unexpected HSI shape: {raw_cube.shape}")
            if not np.isfinite(raw_cube).all():
                raise RuntimeError("The model produced nonfinite spectral values.")
            elapsed = perf_counter() - started
        report(0.9, desc="Preparing spectral views")
        return dict(
            cube=np.clip(raw_cube, 0, 1),
            raw_cube=raw_cube,
            model=key,
            seconds=elapsed,
            original_size=original_size,
            output_size=output_size,
            device=str(self.device),
            ensemble=bool(ensemble),
            processing="Full image",
        )
