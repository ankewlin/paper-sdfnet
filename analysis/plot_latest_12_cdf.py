"""Plot the newest complete, matched seed-42 batch of twelve trained models.

Run from any directory:
    python D:/loc_paper_SDF_net/analysis/plot_latest_12_cdf.py --dry-run
    python D:/loc_paper_SDF_net/analysis/plot_latest_12_cdf.py --device cpu

Training runs default to experiments/<model>/runs_final_prepared/. Use
--runs-dir runs_frozen_config_replay for the selected-configuration rerun batch.
This script evaluates saved best.pt checkpoints on data/prepared.npz and writes a new
CDF directory under analysis/cdf_runs/. It does not train or change checkpoints.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("sdfnet_full", "SDF-Net"),
    ("ablation_mhsa_only", "SDF-Net (MHSA only)"),
    ("ablation_ca_only", "SDF-Net (CA only)"),
    ("ablation_no_ca_mhsa", "SDF-Net (no CA/MHSA)"),
    ("sdfnet_full_all14", "All 11 stats + SAGE"),
    ("ablation_sage_only", "SAGE only"),
    ("ablation_stat_only", "Statistics only"),
    ("single_stream_sdfnet", "Single-stream SDF-Net"),
    ("single_stream_cnn_transformer", "Swin-Loc"),
    ("pure_cnn", "DSCP"),
    ("pure_transformer", "cir-TF"),
    ("sdfnet_rf_rfe", "RF-RFE SDF-Net"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def checked_batch(project, tag, data_hash, runs_dir="runs_final_prepared"):
    runs = {}
    reference_splits = None
    for key, _ in MODELS:
        run = project / "experiments" / key / runs_dir / tag
        for name in ("best.pt", "config.json", "selection_metrics.json", "run.py", "model.py"):
            if not (run / name).is_file():
                raise FileNotFoundError(run / name)
        config = read_json(run / "config.json")
        if config.get("mode") != "train" or config.get("seed") != 42:
            raise ValueError("Unexpected mode or seed: " + str(run))
        if config.get("data_sha256") != data_hash:
            raise ValueError("Training and current prepared.npz differ: " + str(run))
        if not config.get("best_epoch") or "training_seconds" not in config:
            raise ValueError("Training has not completed: " + str(run))
        hashes = config.get("source_hashes") or {}
        for name in ("run.py", "model.py"):
            if hashes.get(name) != sha256(run / name):
                raise ValueError("Saved source differs from training: " + str(run / name))
        splits = config.get("split_ids")
        if not splits or set(splits) != {"train", "val"}:
            raise ValueError("Missing train/validation split records: " + str(run))
        if reference_splits is None:
            reference_splits = splits
        elif splits != reference_splits:
            raise ValueError("Training/validation IDs differ between models: " + str(run))
        if key == "sdfnet_full" and config.get("stat_weighting", "none") != "none":
            raise ValueError("SDF-Net has nonidentity input weighting: " + str(run))
        runs[key] = run
    return runs


def select_latest(project, requested_tag, data_hash, runs_dir="runs_final_prepared"):
    if requested_tag:
        return requested_tag, checked_batch(project, requested_tag, data_hash, runs_dir)
    roots = [project / "experiments" / key / runs_dir for key, _ in MODELS]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    common = set.intersection(*[
        {p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("seed42_")}
        for root in roots
    ])
    failures = []
    for tag in sorted(common, reverse=True):
        try:
            return tag, checked_batch(project, tag, data_hash, runs_dir)
        except (FileNotFoundError, ValueError) as exc:
            failures.append("{}: {}".format(tag, exc))
    raise RuntimeError("No complete matching 12-model batch. " + " | ".join(failures[:3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, help="Default: <project>/data/prepared.npz")
    parser.add_argument("--tag", help="Use one exact seed42_... batch instead of the latest complete one")
    parser.add_argument("--runs-dir", default="runs_final_prepared",
                        help="Training directory under each experiment, e.g. runs_frozen_config_replay")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, help="New plot directory; must not exist")
    parser.add_argument("--dry-run", action="store_true", help="Verify selection without inference")
    args = parser.parse_args()

    if args.runs_dir in ("", ".", "..") or Path(args.runs_dir).name != args.runs_dir:
        parser.error("--runs-dir must be one directory name under each experiment")
    project = args.project.resolve()
    data = (args.data or project / "data" / "prepared.npz").resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    data_hash = sha256(data)
    tag, runs = select_latest(project, args.tag, data_hash, args.runs_dir)
    print("Batch:", tag, "| data SHA256:", data_hash, flush=True)
    for key, label in MODELS:
        print("{}: {}".format(label, runs[key]), flush=True)
    if args.dry_run:
        print("Dry run complete; no output was written.")
        return

    # Reuse the tested inference helpers, including the workaround for Pure CNN's
    # test-mode reader, but supply the new run directories selected above.
    sys.path.insert(0, str(project / "analysis"))
    from plot_cdf_no_right_panel import aligned_errors, device_name, infer_other, infer_pure_cnn

    device = device_name(args.device)
    output = (args.output or project / "analysis" / "cdf_runs" /
              (tag + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference = None
    curves = {}
    records = []
    provenance = {"data": str(data), "data_sha256": data_hash, "tag": tag,
                  "seed": 42, "device": device, "runs_dir": args.runs_dir, "runs": {}}
    for key, label in MODELS:
        run = runs[key]
        checkpoint = run / "best.pt"
        destination = output / "predictions" / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        if key == "pure_cnn":
            ids, truth, pred = infer_pure_cnn(run, checkpoint, data, device, destination)
        else:
            ids, truth, pred = infer_other(project, run, checkpoint, data, device, destination)
        ids, truth, errors = aligned_errors(ids, truth, pred, reference)
        if reference is None:
            reference = ids, truth
        curves[key] = errors
        record = {"model": label, "seed": 42, "n": len(errors),
                  "mean_error_3d": float(errors.mean()),
                  "P50": float(np.median(errors)),
                  "P90": float(np.percentile(errors, 90))}
        records.append(record)
        provenance["runs"][key] = {"path": str(run),
                                   "checkpoint_sha256": sha256(checkpoint)}
        print("{}: mean {:.4f} m (n={})".format(label, errors.mean(), len(errors)), flush=True)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    palette = plt.get_cmap("tab20")
    for index, (key, label) in enumerate(MODELS):
        errors = np.sort(curves[key])
        ax.step(np.r_[0, errors], np.r_[0, np.arange(1, len(errors) + 1) / len(errors)],
                where="post", label="{} (mean: {:.3f} m)".format(label, errors.mean()),
                color=palette(index), linewidth=2.3 if key == "sdfnet_full" else 1.5)
    ax.set(xlabel="3D localization error (m)", ylabel="CDF",
           xlim=(0, None), ylim=(0, 1.01))
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("cdf." + suffix), dpi=240)
    plt.close(fig)
    records.sort(key=lambda row: row["mean_error_3d"])
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (output / "selected_runs.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CDF saved to:", output, flush=True)


if __name__ == "__main__":
    main()
