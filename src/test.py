import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import (
    get_three_way_dataloaders,
    resolve_data_dir,
    check_sha256_duplicates,
)
from model import AgeRegressionModel


def compute_bootstrap_ci(abs_errors, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    boot_maes = [
        float(np.mean(rng.choice(abs_errors, size=len(abs_errors), replace=True)))
        for _ in range(n_boot)
    ]
    low, high = np.percentile(boot_maes, [2.5, 97.5])
    return float(low), float(high)


def evaluate_loader_pytorch(model, loader, device):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for images, ages in loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy().tolist())
            targets.extend(ages.numpy().tolist())
    return np.array(preds, dtype=np.float64), np.array(targets, dtype=np.float64)


def evaluate_loader_onnx(onnx_path, loader):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    preds, targets = [], []
    for images, ages in loader:
        for i in range(images.size(0)):
            inp = images[i : i + 1].numpy()
            out = sess.run(None, {in_name: inp})[0]
            preds.append(float(np.squeeze(out)))
            targets.append(float(ages[i].item()))
    return np.array(preds, dtype=np.float64), np.array(targets, dtype=np.float64)


def test_model(data_dir, weights_path, batch_size=8, use_hybrid=False):
    resolved_dir = resolve_data_dir(data_dir, use_hybrid=use_hybrid)
    valid_files = [
        f
        for f in sorted(glob.glob(os.path.join(resolved_dir, "*.jpg")))
        if " " in os.path.basename(f)
    ]
    total_files, unique_hashes, duplicates = check_sha256_duplicates(valid_files)
    print(
        f"SHA-256 Check ({resolved_dir}): total={total_files}, unique={unique_hashes}, duplicate_groups={len(duplicates)}"
    )
    for digest, names in duplicates.items():
        print(f"  Duplicate SHA-256 {digest[:16]}: {names}")

    train_loader, val_loader, test_loader = get_three_way_dataloaders(
        data_dir, batch_size=batch_size, use_hybrid=use_hybrid
    )

    train_targets = np.array(train_loader.dataset.ages, dtype=np.float64)
    val_targets = np.array(val_loader.dataset.ages, dtype=np.float64)
    test_targets = np.array(test_loader.dataset.ages, dtype=np.float64)

    if weights_path.endswith(".onnx") and os.path.exists(weights_path):
        print(f"Loaded ONNX weights from {weights_path}")
        val_preds, val_targets = evaluate_loader_onnx(weights_path, val_loader)
        test_preds, test_targets = evaluate_loader_onnx(weights_path, test_loader)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = AgeRegressionModel(pretrained=False)
        if os.path.exists(weights_path):
            model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )
            print(f"Loaded PyTorch weights from {weights_path}")
        else:
            print(
                f"Warning: Weights not found at {weights_path}. Evaluating with random weights."
            )
        model.to(device)
        val_preds, val_targets = evaluate_loader_pytorch(model, val_loader, device)
        test_preds, test_targets = evaluate_loader_pytorch(model, test_loader, device)

    val_mae = float(np.mean(np.abs(val_preds - val_targets)))
    test_abs_err = np.abs(test_preds - test_targets)
    test_mae = float(np.mean(test_abs_err))
    ci_low, ci_high = compute_bootstrap_ci(test_abs_err)

    baseline_median_15 = float(np.mean(np.abs(test_targets - 15.0)))
    baseline_train_median = float(
        np.mean(np.abs(test_targets - np.median(train_targets)))
    )
    baseline_train_mean = float(np.mean(np.abs(test_targets - np.mean(train_targets))))

    print(f"Validation MAE (n={len(val_targets)}): {val_mae:.4f} years")
    print(
        f"Independent Test MAE (n={len(test_targets)}): {test_mae:.4f} years (95% Bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}])"
    )
    print(f"Baseline Median (15.0 yrs) Test MAE: {baseline_median_15:.4f} years")
    print(
        f"Baseline Train-Median ({np.median(train_targets):.2f} yrs) Test MAE: {baseline_train_median:.4f} years"
    )
    print(
        f"Baseline Train-Mean ({np.mean(train_targets):.2f} yrs) Test MAE: {baseline_train_mean:.4f} years"
    )

    print("Age-Group Breakdown on Independent Test Set:")
    for g_name, low, high in [
        ("9-11", 9, 11),
        ("12-14", 12, 14),
        ("15-16", 15, 16),
        ("17-19", 17, 19),
    ]:
        mask = (test_targets >= low) & (test_targets <= high)
        g_mae = float(np.mean(np.abs(test_preds[mask] - test_targets[mask])))
        print(f"  Age {g_name}: n={int(np.sum(mask))}, MAE={g_mae:.4f} years")

    plt.figure(figsize=(8, 6))
    plt.scatter(test_targets, test_preds, alpha=0.6)
    plt.plot(
        [min(test_targets), max(test_targets)],
        [min(test_targets), max(test_targets)],
        "r--",
    )
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Independent Test Age Estimation (MAE: {test_mae:.2f} yrs)")
    plt.grid(True)
    plt.savefig("evaluation_results.png")
    print("Saved plot to evaluation_results.png")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "dataset ++")
    CANDIDATE_WEIGHTS = [
        os.path.join(BASE_DIR, "src", "weights", "best_hybrid_age_model.pth"),
        os.path.join(BASE_DIR, "src", "weights", "best_hybrid_age_model.onnx"),
        os.path.join(
            BASE_DIR,
            "..",
            "Emergency_Weights_Vault",
            "Dental_011",
            "weights",
            "best_hybrid_age_model.onnx",
        ),
        os.path.join(BASE_DIR, "src", "weights", "best_age_model.pth"),
    ]
    WEIGHTS_PATH = next(
        (p for p in CANDIDATE_WEIGHTS if os.path.exists(p)), CANDIDATE_WEIGHTS[0]
    )
    test_model(DATA_DIR, WEIGHTS_PATH, batch_size=8, use_hybrid=True)

