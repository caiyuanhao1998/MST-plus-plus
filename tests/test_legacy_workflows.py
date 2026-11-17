"""Regressions for the README's training, metrics and submission workflows."""

import ast
import importlib.util
import io
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5storage
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("directory", ["train_code", "test_develop_code"])
def test_metrics_preserve_noncontiguous_pixel_order_and_gradients(directory):
    utils = load_source("metrics_" + directory, directory + "/utils.py")
    generator = torch.Generator().manual_seed(17)
    target = (torch.rand((2, 31, 8, 10), generator=generator) * 0.8 + 0.1).transpose(2, 3)
    output = (target * 0.7).detach().requires_grad_()
    assert not output.is_contiguous()
    difference = output - target
    expected = [
        difference.abs().div(target).mean(),
        difference.square().mean().sqrt(),
        (-10 * torch.log10(difference.square().mean(dim=(1, 2, 3)))).mean(),
    ]
    for loss_type, reference in zip([utils.Loss_MRAE, utils.Loss_RMSE, utils.Loss_PSNR], expected):
        actual = loss_type()(output, target)
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual)
    utils.Loss_MRAE()(output, target).backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


def test_submission_uses_current_mat73_api_and_preserves_center_crop(tmp_path):
    module = load_source("submission", "test_challenge_code/prep_submission.py")
    source = tmp_path / "input"
    source.mkdir()
    cube = np.broadcast_to(np.linspace(0.1, 0.9, 31, dtype=np.float32), (482, 512, 31)).copy()
    hdf5storage.savemat(str(source / "scene.mat"), {"cube": cube}, format="7.3")
    destination = tmp_path / "submission"
    module.main(["-i", str(source), "-o", str(destination)])
    with zipfile.ZipFile(destination / "submission.zip") as archive:
        assert archive.namelist() == ["scene_crop.mat"]
        with h5py.File(io.BytesIO(archive.read("scene_crop.mat"))) as mat:
            recovered = np.array(mat["cube"]).T
    assert recovered.shape == (226, 256, 31)
    np.testing.assert_array_equal(recovered, cube[128:-128, 128:-128])


def test_submission_rejects_empty_input(tmp_path):
    module = load_source("empty_submission", "test_challenge_code/prep_submission.py")
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="No reconstructed MAT"):
        module.main(["-i", str(source), "-o", str(tmp_path / "output")])


def test_training_finishes_current_loader_before_stopping(monkeypatch, tmp_path):
    # Load the real legacy loop without its import-time dataset and CUDA setup.
    parsed = ast.parse((ROOT / "train_code/train.py").read_text())
    function = next(
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    model = nn.Conv2d(3, 31, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
    data = TensorDataset(torch.full((5, 3, 8, 8), 0.5), torch.full((5, 31, 8, 8), 0.5))
    steps, saved, validations = [], [], []
    optimizer.register_step_post_hook(lambda *args, **kwargs: steps.append(1))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    def loader(dataset, **kwargs):
        kwargs.pop("num_workers", None)
        kwargs.pop("pin_memory", None)
        return DataLoader(dataset, num_workers=0, **kwargs)

    def validate(*args):
        validations.append(1)
        return torch.tensor(0.2), torch.tensor(0.1), torch.tensor(30.0)

    namespace = {
        "torch": torch,
        "cudnn": torch.backends.cudnn,
        "Variable": torch.autograd.Variable,
        "total_iteration": 2,
        "model": model,
        "train_data": data,
        "val_data": data,
        "DataLoader": loader,
        "opt": SimpleNamespace(batch_size=1, outf=str(tmp_path)),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion_mrae": lambda output, target: (output - target).abs().mean(),
        "AverageMeter": load_source("train_metrics", "train_code/utils.py").AverageMeter,
        "validate": validate,
        "save_checkpoint": lambda path, epoch, iteration, *args: saved.append(iteration),
        "logger": logging.getLogger("training-loop-regression"),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<real training loop>", "exec"),
        namespace,
    )
    assert namespace["main"]() == 0
    # The limit is checked between complete DataLoader passes, as intended.
    assert len(steps) == len(data) == 5
    assert validations == []
    assert saved == []


@pytest.mark.parametrize(
    "method,expected_input", [("awan", (246, 276)), ("mst_plus_plus", (482, 512))]
)
def test_training_validation_uses_the_documented_center_region(monkeypatch, method, expected_input):
    parsed = ast.parse((ROOT / "train_code/train.py").read_text())
    function = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate"
    )
    seen, compared = [], []
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    class Model(nn.Module):
        def forward(self, tensor):
            seen.append(tuple(tensor.shape[-2:]))
            return tensor[:, :1].expand(-1, 31, -1, -1)

    def metric(output, target):
        assert output.shape == target.shape
        compared.append(tuple(output.shape[-2:]))
        return (output - target).abs().mean()

    namespace = {
        "torch": torch,
        "opt": SimpleNamespace(method=method),
        "AverageMeter": load_source("validation_metrics", "train_code/utils.py").AverageMeter,
        "criterion_mrae": metric,
        "criterion_rmse": metric,
        "criterion_psnr": metric,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<real validation loop>", "exec"),
        namespace,
    )
    inputs = [(torch.full((1, 3, 482, 512), 0.5), torch.full((1, 31, 482, 512), 0.7))]
    model = Model()
    results = namespace["validate"](inputs, model)
    assert model.training, "Validation must restore training mode"
    assert seen == [expected_input]
    assert compared == [(226, 256)] * 3
    assert all(torch.isfinite(value) for value in results)
