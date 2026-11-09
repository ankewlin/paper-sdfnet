"""Evaluate saved RF-RFE SDF-Net checkpoints on the complete existing test split."""
import argparse
import csv
from pathlib import Path

import numpy as np

from train import HERE, import_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    shared = import_file("path_count_shared", HERE.parent / "sdfnet_path_count/_shared.py")
    study = args.study.resolve()
    manifest = shared.read_json(study / "study.json")
    if not manifest.get("complete"):
        raise ValueError("All requested training seeds must finish before final test evaluation.")
    data = Path(manifest["data"])
    if shared.sha256(data) != manifest["data_sha256"]:
        raise ValueError("Prepared dataset changed.")
    output = study / ("test_" + shared.stamp())
    output.mkdir(exist_ok=False)
    rows, ids = [], None
    for seed in manifest["seeds"]:
        run = study / ("seed_{}".format(seed))
        shared.check_run(run, seed, 15, manifest["data_sha256"])
        if shared.read_json(run / "config.json")["source_hashes"] != manifest["source_hashes"]:
            raise ValueError("Saved training source differs from this study.")
        destination = output / ("seed_{}".format(seed))
        command = [__import__("sys").executable, "-u", str(run / "run.py"),
                   "--mode", "test", "--checkpoint", str(run / "best.pt"),
                   "--data", str(data), "--device", args.device, "--output", str(destination)]
        shared.execute(command)
        metrics = shared.read_json(destination / "metrics.json")
        config = shared.read_json(destination / "config.json")
        current_ids = config["split_ids"]["test"]
        if ids is not None and ids != current_ids:
            raise ValueError("Test positions differ across seeds.")
        ids = current_ids
        rows.append(dict(seed=seed, n=metrics["n"], mean_error=metrics["mean_error_3d"],
                         p50=metrics["median_error_3d"], p90=metrics["p90_error_3d"]))
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = np.array([row["mean_error"] for row in rows])
    summary = dict(per_seed=rows, mean_error=float(means.mean()),
                   std_across_run_means=float(means.std(ddof=0)), std_ddof=0,
                   p50_mean_across_runs=float(np.mean([row["p50"] for row in rows])),
                   p90_mean_across_runs=float(np.mean([row["p90"] for row in rows])),
                   test_positions=len(ids), excluded_test_positions=[],
                   note="Complete prepared test split. Do not compare directly to metrics on a smaller plotting subset.")
    shared.write_json(output / "summary.json", summary)
    print("Full test: n={}, mean={:.4f} +/- {:.4f} m".format(len(ids), means.mean(), means.std(ddof=0)))
    print("Saved per-position predictions and CDF data: " + str(output))


if __name__ == "__main__":
    main()
