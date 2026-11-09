import os
import sys
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for cand in (ROOT_DIR, SRC_DIR):
    if cand not in sys.path:
        sys.path.insert(0, cand)

from dataset import DentalAgeDataset, check_sha256_duplicates, split_dataset_files
from model import AgeRegressionModel
from test import compute_bootstrap_ci, evaluate_loader_pytorch


def test_age_regression_model_smoke_forward():
    model = AgeRegressionModel(pretrained=False)
    model.eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_evaluate_loader_pytorch_and_bootstrap_ci():
    model = AgeRegressionModel(pretrained=False)
    ds = TensorDataset(
        torch.zeros(4, 3, 224, 224),
        torch.tensor([10.0, 12.0, 15.0, 18.0], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=2)
    preds, targets = evaluate_loader_pytorch(model, loader, torch.device("cpu"))
    assert preds.shape == (4,)
    assert targets.shape == (4,)

    abs_err = np.abs(preds - targets)
    ci_low, ci_high = compute_bootstrap_ci(abs_err, n_boot=200, seed=42)
    assert 0.0 <= ci_low <= ci_high


def test_dataset_filename_parsing_and_three_way_split(tmp_path):
    sample_files = []
    for idx in range(10):
        fp = tmp_path / f"{10 + idx} {idx}.jpg"
        img = Image.fromarray(np.full((32, 32, 3), idx * 20, dtype=np.uint8))
        img.save(str(fp))
        sample_files.append(str(fp))

    ds = DentalAgeDataset(sample_files[:2])
    assert len(ds) == 2
    assert ds.ages == [10.0, 11.0]

    total, unique_cnt, dupes = check_sha256_duplicates(sample_files)
    assert total == 10
    assert unique_cnt == 10
    assert len(dupes) == 0

    train_f, val_f, test_f = split_dataset_files(sample_files, random_state=42)
    assert len(train_f) + len(val_f) + len(test_f) == 10
    assert set(train_f).isdisjoint(set(val_f))
    assert set(train_f).isdisjoint(set(test_f))
    assert set(val_f).isdisjoint(set(test_f))
