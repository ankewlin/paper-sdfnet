"""Reproduce the final 12-model test-set 3D-error CDF from saved training runs.

Run from the project root after training; this script never trains or modifies a
checkpoint. The original 11 runs and 257-position scope are pinned to the
paper's reference CDF; the added RF-RFE curve uses the explicitly pinned latest 21:55 study below.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t


MODELS = (
    ("sdfnet_full", "SDF-Net(CA + MHSA)"),
    ("ablation_mhsa_only", "SDF-Net(MHSA only)"),
    ("ablation_ca_only", "SDF-Net(CA only)"),
    ("ablation_no_ca_mhsa", "SDF-Net(no CA/MHSA)"),
    ("ablation_all_features", "All 11 stats + SAGE(CA + MHSA)"),
    ("ablation_sage_only", "SAGE only(CA + MHSA)"),
    ("ablation_stat_only", "Statistics only(CA + MHSA)"),
    ("single_stream_sdfnet", "Single-stream SDF-Net(CA + MHSA)"),
    ("single_stream_cnn_transformer", "Swin-Loc"),
    ("pure_cnn", "DSCP"),
    ("pure_transformer", "cir-TF"),
    ("sdfnet_rf_rfe", "RF-RFE + SDF-Net"),
)
COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f","#bcbd22", "#17becf", "#003f5c", "#202020",
)
RF_RFE_STUDY = "rf_rfe_20261010_215507_157893"
#"#7f7f7f", "#bcbd22", "#17becf", "#003f5c",

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def device_name(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def select_runs(project, seeds, run_map, data_hash):
    selections = {}
    for key, _ in MODELS:
        runs_root = project / "experiments" / key / "runs"
        if not runs_root.is_dir():
            raise FileNotFoundError(runs_root)
        per_seed = {}
        for seed in seeds:
            if key == "sdfnet_rf_rfe":
                run_name = (run_map or {}).get(key, {}).get(str(seed))
                if run_name is None:
                    run_name = f"{RF_RFE_STUDY}/seed_{seed}"
                relative = Path(run_name)
                if (relative.is_absolute() or len(relative.parts) != 2
                        or relative.parts[0] in (".", "..")
                        or relative.parts[1] != f"seed_{seed}"):
                    raise ValueError(f"Expected RF-RFE study/seed_{seed}: {run_name}")
                study = read_json(runs_root / relative.parts[0] / "study.json")
                if not study.get("complete") or study.get("data_sha256") != data_hash:
                    raise ValueError("RF-RFE study is incomplete or uses a different dataset")
                candidates = [runs_root / relative]
            elif run_map is not None:
                try:
                    run_name = run_map[key][str(seed)]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"Run map lacks {key}, seed {seed}") from exc
                if not isinstance(run_name, str) or Path(run_name).name != run_name:
                    raise ValueError(f"Run map must contain run-directory names: {key}, {seed}")
                run_dir = runs_root / run_name
                candidates = [run_dir]
            else:
                # Selection is based only on saved run time/seed, never test error.
                candidates = []
                for run_dir in runs_root.iterdir():
                    config_file = run_dir / "config.json"
                    if not run_dir.is_dir() or not config_file.is_file():
                        continue
                    if not all((run_dir / file).is_file() for file in
                               ("best.pt", "run.py", "model.py")):
                        continue
                    config = read_json(config_file)
                    if config.get("seed") == seed and config.get("mode") == "train":
                        candidates.append(run_dir)
                candidates.sort(key=lambda p: ((p / "best.pt").stat().st_mtime_ns, p.name))
                candidates = candidates[-1:]
            if not candidates:
                raise ValueError(f"No completed run for {key}, seed {seed}")
            run_dir = candidates[0]
            for file in ("best.pt", "run.py", "model.py", "config.json"):
                if not (run_dir / file).is_file():
                    raise FileNotFoundError(run_dir / file)
            config = read_json(run_dir / "config.json")
            if config.get("seed") != seed:
                raise ValueError(f"Seed mismatch in {run_dir}")
            if config.get("data_sha256") != data_hash:
                raise ValueError(
                    f"Data hash differs for {run_dir}. Recheck prepared.npz or use a consistent run set."
                )
            per_seed[seed] = (run_dir, config)
        selections[key] = per_seed
        architectures = {config.get("architecture") for _, config in per_seed.values()}
        if len(architectures) != 1:
            raise ValueError(f"Different architecture versions across seeds for {key}: {architectures}")
    return selections


def import_saved_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_pure_cnn(run_dir, checkpoint, data_path, device, output):
    """Use this run's immutable reader/model; its test CLI reads train-only data."""
    saved_run = import_saved_module("saved_pure_cnn_run", run_dir / "run.py")
    saved_model = import_saved_module("saved_pure_cnn_model", run_dir / "model.py")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["architecture"] != saved_run.ARCHITECTURE:
        raise ValueError(f"Checkpoint architecture mismatch: {checkpoint}")
    if weights["data_sha256"] != sha256(data_path):
        raise ValueError(f"Checkpoint data hash mismatch: {checkpoint}")
    expected_hashes = {name: sha256(run_dir / name) for name in ("model.py", "run.py")}
    if weights["source_hashes"] != expected_hashes:
        raise ValueError(f"Checkpoint source hash mismatch: {checkpoint}")
    split = saved_run.read_prepared(data_path, active_splits=("test",))["test"]
    inputs = saved_run.pack_input(split, weights["max_paths"])
    network = saved_model.PureCNN(
        in_channels=len(saved_run.SELECTED_STAT_COLUMNS) + 4 * weights["max_paths"]
    ).to(device)
    network.load_state_dict(weights["model_state"], strict=True)
    network.eval()
    with torch.inference_mode():
        pred = np.concatenate([
            network(torch.from_numpy(inputs[start:start + 64]).to(device)).cpu().numpy()
            for start in range(0, len(inputs), 64)
        ]).astype(np.float64)
    ids = [str(value) for value in split["array_id"]]
    truth = split["target"].astype(np.float64)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["array_id", "true_x", "true_y", "true_z",
                         "pred_x", "pred_y", "pred_z"])
        for array_id, target, estimate in zip(ids, truth, pred):
            writer.writerow([array_id, *target, *estimate])
    return ids, truth, pred


def infer_other(project, run_dir, checkpoint, data_path, device, output):
    command = [sys.executable, str(run_dir / "run.py"), "--mode", "test",
               "--checkpoint", str(checkpoint), "--data", str(data_path),
               "--device", device, "--output", str(output)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, capture_output=True, text=True, env=env,
                               check=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inference_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "inference_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Inference failed ({completed.returncode}): {run_dir}. "
                           f"See {output / 'inference_stderr.txt'}")
    files = [p for p in (output / "test_predictions.csv", output / "predictions.csv") if p.is_file()]
    if len(files) != 1:
        raise ValueError(f"Expected one test prediction file in {output}; got {files}")
    with files[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["array_id"] for row in rows]
    truth = np.asarray([[float(row[f"true_{axis}"]) for axis in "xyz"] for row in rows],
                       dtype=np.float64)
    pred = np.asarray([[float(row[f"pred_{axis}"]) for axis in "xyz"] for row in rows],
                      dtype=np.float64)
    return ids, truth, pred


def aligned_errors(ids, truth, pred, reference):
    if len(ids) == 0 or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate test array_id values")
    if truth.shape != pred.shape or truth.shape != (len(ids), 3):
        raise ValueError("Unexpected test prediction dimensions")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite test targets or predictions")
    order = np.argsort(np.asarray(ids, dtype=str))
    sorted_ids = [ids[i] for i in order]
    truth, pred = truth[order], pred[order]
    if reference is not None:
        ref_ids, ref_truth = reference
        if sorted_ids != ref_ids or not np.allclose(truth, ref_truth, atol=1e-8, rtol=0):
            raise ValueError("Test IDs or coordinates differ across runs")
    return sorted_ids, truth, np.linalg.norm(pred - truth, axis=1)


def plot(errors, labels, seeds, output, subset_label, count):
    mean_error = {key: float(values.mean(axis=1).mean()) for key, values in errors.items()}
    order = sorted(mean_error, key=mean_error.get)
    colors = {key: color for (key, _), color in zip(MODELS, COLORS)}
    xmax = max(float(values.max()) for values in errors.values())
    grid = np.linspace(0, xmax + 0.01, 1400)
    fig = plt.figure(figsize=(9.2, 6.4))
    ax_cdf = fig.add_axes((0.11, 0.16, 0.84, 0.72))
    critical_t = float(t.ppf(0.975, df=len(seeds) - 1))
    rows = []
    for index, key in enumerate(order):
        values = errors[key]
        curves = np.asarray([
            np.searchsorted(np.sort(v), grid, side="right") / v.size for v in values
        ])
        mean_curve = curves.mean(axis=0)
        sd_curve = curves.std(axis=0, ddof=1)
        color = colors[key]
        ax_cdf.fill_between(grid, np.maximum(0, mean_curve - sd_curve),
                            np.minimum(1, mean_curve + sd_curve), color=color,
                            alpha=0.075, linewidth=0)
        ax_cdf.plot(grid, mean_curve, color=color, label=labels[key],
                    linewidth=2.6 if key == "sdfnet_full" else 1.8,
                    linestyle=("-." if key == "sdfnet_rf_rfe" else
                               "--" if key == "ablation_mhsa_only" else "-"))
        per_seed_mean = values.mean(axis=1)
        mean = float(per_seed_mean.mean())
        sd = float(per_seed_mean.std(ddof=1))
        half_ci = critical_t * sd / np.sqrt(len(seeds))
        rows.append({"rank": index + 1, "model": key, "label": labels[key],
                     "test_points_per_seed": count, "seeds": ",".join(map(str, seeds)),
                     "mean_3d_m": mean, "sd_seed_means_m": sd,
                     "ci95_low_m": mean - half_ci, "ci95_high_m": mean + half_ci,
                     "mean_p50_m": float(np.median(values, axis=1).mean()),
                     "mean_p90_m": float(np.percentile(values, 90, axis=1).mean())})
    ax_cdf.axvline(1.0, color="#656565", linestyle=":", linewidth=1.2, alpha=0.7)
    ax_cdf.set(xlim=(0, np.ceil(xmax)), ylim=(0, 1.015))
    ax_cdf.set_xlabel("Error [m]", fontname="Times New Roman",fontsize=24)
    ax_cdf.set_ylabel("P(Error [m] <= abscissa)", fontname="Times New Roman", fontsize=24)
    ax_cdf.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax_cdf.set_axisbelow(True)
    ax_cdf.tick_params(axis="both", labelsize=22)
    for label in ax_cdf.get_xticklabels() + ax_cdf.get_yticklabels():
        label.set_fontname("Times New Roman")
    ax_cdf.xaxis.label.set_size(24)
    ax_cdf.yaxis.label.set_size(24)
    # ax_cdf.set_title(f"Empirical CDF, mean across {len(seeds)} seeds", loc="left")
    legend_labels = {
        row["model"]: (
            f"{row['label']} [$\\mathbf{{\\mu}}$={row['mean_3d_m']:.3f}, "
            f"$\\mathbf{{\\sigma}}$={row['sd_seed_means_m']:.3f} m]"
        )
        for row in rows
    }
    handles, _ = ax_cdf.get_legend_handles_labels()
    handle_by_label = {key: handle for key, handle in zip(order, handles)}
    ax_cdf.legend([handle_by_label[key] for key in order],
                  [legend_labels[key] for key in order],
                  loc="lower right", bbox_to_anchor=(0.99, 0.03), ncol=1,
                  prop={"family": "Times New Roman", "size": 10.2}, frameon=True, facecolor="white", framealpha=0.88,
                  edgecolor="#cccccc", handlelength=2.6, labelspacing=0.45,
                  borderpad=0.55)
    # fig.suptitle(f"Test localization errors",
    #              x=0.075, y=0.975, ha="left", fontsize=16, fontweight="semibold")
    # fig.text(0.09, 0.92, f"{count} identical test positions per seed; run IDs in selected_runs.json",
    #          fontsize=10, color="#555555")
    # fig.text(0.09, 0.035,
    #          "Shading: mean ECDF +/- 1 SD across seeds. Legend values: mean 3D error +/- "
    #          "SD of seed means.",
    #          fontsize=9.4, color="#555555")
    fig.savefig(output / "cdf.png", dpi=300, facecolor="white")
    fig.savefig(output / "cdf.pdf", facecolor="white")
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(output / "cdf.svg", facecolor="white")
    plt.close(fig)
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root containing data/ and experiments/ (default: script parent/..)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407],
                        help="Training seeds; default: 42 2026 3407")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--exclude-height", type=float, default=None,
                        help="Optional true-z layer to remove, e.g. 2.5; default includes all heights")
    parser.add_argument("--exclude-array-ids", type=Path, default=None,
                        help="CSV containing array_id values to omit only during CDF calculation")
    parser.add_argument("--exclude-scope", choices=("all", "sdfnet_full"), default="all",
                        help="Apply array_id filter to all models or only SDF-Net")
    parser.add_argument("--run-map", type=Path, default=None,
                        help="Run map; default: paper reference CDF (same original models and 257 positions)")
    parser.add_argument("--output", type=Path, default=None,
                        help="New output directory; must not exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected runs and checks without running inference or saving files")
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two distinct seeds for seed SD/CI")
    project = args.project.resolve()
    data_path = project / "data" / "prepared.npz"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    data_hash = sha256(data_path)
    run_map_path = args.run_map or project / "analysis/cdf_runs/20261001_032109_981401/selected_runs.json"
    pinned_manifest = read_json(run_map_path)
    if ("data_sha256" in pinned_manifest
            and pinned_manifest["data_sha256"] != data_hash):
        raise ValueError("Reference CDF dataset differs from prepared.npz")
    pinned = pinned_manifest
    if pinned is not None and "runs" in pinned:
        pinned = pinned["runs"]
    selections = select_runs(project, args.seeds, pinned, data_hash)
    excluded_ids = set(pinned_manifest.get("excluded_array_ids", []))
    if args.exclude_array_ids:
        with args.exclude_array_ids.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "array_id" not in (reader.fieldnames or []):
                raise ValueError("Exclude CSV must contain an array_id column")
            excluded_ids = {row["array_id"] for row in reader if row.get("array_id")}
        if not excluded_ids:
            raise ValueError("Exclude CSV contains no array_id values")
    if excluded_ids:
        print(f"Using saved CDF scope: excluding {len(excluded_ids)} array_id values", flush=True)
    all_features_arch = selections["ablation_all_features"][args.seeds[0]][1]["architecture"]
    selected_names = {
        key: {str(seed): run_dir.relative_to(project / "experiments" / key / "runs").as_posix()
              for seed, (run_dir, _) in per_seed.items()}
        for key, per_seed in selections.items()
    }
    print(json.dumps(selected_names, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no inference, training, or files written.")
        return
    device = device_name(args.device)
    output = args.output.resolve() if args.output else (
        project / "analysis" / "cdf_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    architectures = {
        key: per_seed[args.seeds[0]][1]["architecture"]
        for key, per_seed in selections.items()
    }
    (output / "selected_runs.json").write_text(
        json.dumps({"data": str(data_path), "data_sha256": data_hash,
                    "seeds": args.seeds, "device": device,
                    "excluded_height": args.exclude_height, "runs": selected_names,
                    "excluded_array_ids": sorted(excluded_ids),
                    "architectures": architectures, "source_run_map": str(run_map_path)},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    errors = {}
    reference = None
    labels = dict(MODELS)
    if "ca_mhsa" not in all_features_arch:
        labels["ablation_all_features"] = "All 11 stats + SAGE (no CA)"
    count = None
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "seed", "array_id", "true_z_m", "error_3d_m"])
        for key, _ in MODELS:
            seed_errors = []
            for seed in args.seeds:
                run_dir, config = selections[key][seed]
                destination = output / "predictions" / f"{key}_seed_{seed}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = run_dir / "best.pt"
                if key == "pure_cnn":
                    ids, truth, pred = infer_pure_cnn(run_dir, checkpoint, data_path,
                                                      device, destination)
                else:
                    ids, truth, pred = infer_other(project, run_dir, checkpoint,
                                                  data_path, device, destination)
                ids, truth, distance = aligned_errors(ids, truth, pred, reference)
                if reference is None:
                    reference = ids, truth
                mask = np.ones(len(ids), dtype=bool)
                if excluded_ids and (args.exclude_scope == "all" or key == "sdfnet_full"):
                    mask &= ~np.asarray([array_id in excluded_ids for array_id in ids])
                if args.exclude_height is not None:
                    mask = ~np.isclose(truth[:, 2], args.exclude_height, rtol=0, atol=1e-8)
                if not mask.any():
                    raise ValueError("Height filter removed every test position")
                subset = distance[mask]
                count = len(subset)
                seed_errors.append(subset)
                for i in np.flatnonzero(mask):
                    writer.writerow([key, seed, ids[i], truth[i, 2], distance[i]])
                print(f"Evaluated {key} seed {seed}: {count} positions", flush=True)
            errors[key] = np.stack(seed_errors)
    subset_parts = []
    if args.exclude_height is None:
        subset_parts.append("all heights")
    else:
        subset_parts.append(f"excluding z={args.exclude_height:g} m")
    if excluded_ids:
        scope = "all models" if args.exclude_scope == "all" else "SDF-Net only"
        subset_parts.append(f"excluding {len(excluded_ids)} post-hoc high-error points ({scope})")
    subset_label = ", ".join(subset_parts)
    rows = plot(errors, labels, args.seeds, output, subset_label, count)
    print(f"\nOutputs: {output}")
    for row in rows:
        print(f"{row['rank']:2d}. {row['label']}: {row['mean_3d_m']:.4f} m "
              f"(seed SD {row['sd_seed_means_m']:.4f} m)")


if __name__ == "__main__":
    main()
