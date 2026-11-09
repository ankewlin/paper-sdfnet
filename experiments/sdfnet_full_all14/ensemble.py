"""Reproduce fixed SDF-Net ensembles from the completed overnight search.

Validation is the default. Use --mode test only after choosing one recipe.
No ensemble weights are fitted to validation labels.
"""
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from model import SDFNet
from run import ARCHITECTURE, PROJECT, metrics, pack_input, read_prepared, sha256


HERE = Path(__file__).resolve().parent
DEFAULT_SEARCH = HERE / "searches" / "overnight_20260926_014205_739376"
SEEDS = (42, 2026, 3407, 9091)
RECIPES = {
    "best4_mean": ("mean", tuple((8, seed) for seed in SEEDS)),
    "top3_median": ("median", tuple((candidate, seed)
                                    for candidate in (8, 13, 63) for seed in SEEDS)),
}
EXPECTED_CONFIG = {
    8: {"lr": 0.00096194, "feature_noise": 0.05, "ema_decay": 0.0,
        "d_model": 96, "num_layers": 2},
    13: {"lr": 0.00040759, "feature_noise": 0.03, "ema_decay": 0.0,
         "d_model": 96, "num_layers": 2},
    63: {"lr": 0.00117374, "feature_noise": 0.03, "ema_decay": 0.995,
         "d_model": 64, "num_layers": 1},
}


def saved_validation(search_dir, candidate, seed, expected_ids, expected_target):
    path = search_dir / "candidate_{:03d}_seed_{}".format(candidate, seed) / "validation_predictions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["array_id"] for row in rows]
    target = np.asarray([[float(row["true_" + axis]) for axis in "xyz"] for row in rows])
    pred = np.asarray([[float(row["pred_" + axis]) for axis in "xyz"] for row in rows])
    if ids != list(expected_ids) or not np.allclose(target, expected_target, atol=1e-6, rtol=0):
        raise ValueError("Saved validation IDs/targets differ: " + str(path))
    return pred


def predict_member(checkpoint_path, split, data_hash, code_hashes, device, batch_size):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != ARCHITECTURE or checkpoint["data_sha256"] != data_hash:
        raise ValueError("Architecture or data mismatch: " + str(checkpoint_path))
    if checkpoint["source_hashes"] != code_hashes:
        raise ValueError("Model/runner code differs from checkpoint: " + str(checkpoint_path))
    packed = pack_input(split, checkpoint["input_config"])
    dataset = TensorDataset(torch.from_numpy(packed["statistical"]),
                            torch.from_numpy(packed["physical"]),
                            torch.from_numpy(packed["path_mask"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = SDFNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    result = []
    with torch.no_grad():
        for stat, physical, mask in loader:
            result.append(model(stat.to(device), physical.to(device), mask.to(device)).cpu().numpy())
    return np.concatenate(result), checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "prepared.npz")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="best4_mean")
    parser.add_argument("--mode", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, help="New output directory")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    import torch
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        parser.error("Choose an available CPU or CUDA device")
    combine, members = RECIPES[args.recipe]
    search_dir = args.search_dir.resolve()
    with (search_dir / "plan.json").open(encoding="utf-8") as handle:
        plan = json.load(handle)
    data_hash = sha256(args.data)
    if plan["data_sha256"] != data_hash or plan["run_sha256"] != sha256(HERE / "run.py") \
            or plan["model_sha256"] != sha256(HERE / "model.py"):
        raise ValueError("Search plan data/model/runner differs from current files")
    for candidate, _ in members:
        actual = plan["candidates"][candidate - 1]
        if any(actual[key] != value for key, value in EXPECTED_CONFIG[candidate].items()):
            raise ValueError("Search plan does not contain the expected fixed recipe")
    code_hashes = {name: sha256(HERE / name) for name in ("run.py", "model.py")}
    split_name = "val" if args.mode == "validation" else "test"
    splits, _ = read_prepared(args.data, (split_name,))
    split = splits[split_name]
    ids, target = split["array_id"], split["target"].astype(np.float64)
    predictions = []
    details = []
    for candidate, seed in members:
        run_dir = search_dir / "candidate_{:03d}_seed_{}".format(candidate, seed)
        checkpoint_path = run_dir / "best.pt"
        pred, checkpoint = predict_member(checkpoint_path, split, data_hash, code_hashes,
                                          device, args.batch_size)
        if args.mode == "validation":
            expected = saved_validation(search_dir, candidate, seed, ids, target)
            # CPU and CUDA inference kernels can differ slightly across PyTorch builds.
            discrepancy = np.abs(pred - expected)
            if discrepancy.mean() > 0.005 or discrepancy.max() > 0.05:
                raise ValueError("Checkpoint predictions differ from saved validation CSV: {} | max {:.6g}, mean {:.6g}".format(
                    run_dir, discrepancy.max(), discrepancy.mean()))
        predictions.append(pred.astype(np.float64))
        details.append(dict(candidate=candidate, seed=seed, checkpoint=str(checkpoint_path),
                            checkpoint_sha256=sha256(checkpoint_path),
                            epoch=checkpoint["epoch"],
                            validation_mse=checkpoint["validation"]["coordinate_mse"]))
        print("Loaded candidate {} seed {}".format(candidate, seed), flush=True)
    stack = np.stack(predictions)
    combined = stack.mean(axis=0) if combine == "mean" else np.median(stack, axis=0)
    disagreement = np.linalg.norm(stack - combined[None], axis=2).mean(axis=0)
    result = metrics(combined, target)
    result["per_height"] = {str(z): metrics(combined[target[:, 2] == z],
                                            target[target[:, 2] == z])
                            for z in np.unique(target[:, 2])}
    result.update(recipe=args.recipe, combination=combine, split=split_name,
                  data_sha256=data_hash, device=str(device), members=details,
                  validation_selected=(args.mode == "validation"))
    output = args.output or HERE / "ensembles" / (args.mode + "_" + args.recipe + "_" +
                                                   datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / (split_name + "_predictions.csv")).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["array_id", "true_x", "true_y", "true_z", "pred_x", "pred_y",
                         "pred_z", "error_3d", "mean_member_disagreement_3d"])
        error = np.linalg.norm(combined - target, axis=1)
        for sample_id, truth, pred, distance, spread in zip(ids, target, combined, error, disagreement):
            writer.writerow([sample_id] + truth.tolist() + pred.tolist() + [distance, spread])
    shutil.copy2(HERE / "ensemble.py", output / "ensemble.py")
    print("{} {} | coordinate MSE {:.6f} | mean 3D error {:.4f} | {}".format(
        args.recipe, split_name, result["coordinate_mse"], result["mean_error_3d"], output), flush=True)


if __name__ == "__main__":
    main()
