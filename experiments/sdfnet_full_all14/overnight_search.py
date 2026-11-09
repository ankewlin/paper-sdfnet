"""Validation-only overnight search for the CA + MHSA SDF-Net.

Screen many configurations with one fixed seed, then confirm the best ones
with three NEW seeds. Trials run sequentially on one device. No test split is read.
The plan and each completed result are saved so --resume can continue later.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run import PROJECT, ARCHITECTURE


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
BASELINE = dict(lr=0.001, weight_decay=0.0001, dropout=0.1,
                feature_noise=0.0, ema_decay=0.98, aux_weight=0.1,
                distance_weight=0.1, max_paths=15, angle_encoding="circular",
                d_model=96, num_layers=2, cnn_width=32, cnn_blocks=2)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def sample(rng, mixed):
    candidate = dict(BASELINE)
    candidate.update(
        lr=round(math.exp(rng.uniform(math.log(2e-4), math.log(1.2e-3))), 8),
        weight_decay=(0.0 if rng.random() < 0.1 else
                      round(math.exp(rng.uniform(math.log(1e-6), math.log(1e-2))), 8)),
        dropout=rng.choice([0.1, 0.2, 0.3, 0.4]),
        feature_noise=rng.choice([0.0, 0.01, 0.03, 0.05]),
        ema_decay=rng.choice([0.0, 0.98, 0.995]),
        aux_weight=rng.choice([0.0, 0.05, 0.1, 0.2]),
        distance_weight=rng.choice([0.0, 0.1, 0.2]),
    )
    if mixed:
        candidate.update(
            max_paths=rng.choice([5, 8, 12, 15, 15]),
            angle_encoding=rng.choice(["circular", "circular", "scalar"]),
            d_model=rng.choice([64, 96, 128]),
            num_layers=rng.choice([1, 2, 3]),
            cnn_width=rng.choice([24, 32, 48]),
            cnn_blocks=rng.choice([1, 2]),
        )
    return candidate


def make_candidates(count, seed):
    rng = random.Random(seed)
    result = [dict(BASELINE)]
    seen = {json.dumps(result[0], sort_keys=True)}
    regularization_count = (count - 1) // 2
    while len(result) < count:
        candidate = sample(rng, mixed=len(result) > regularization_count)
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def read_records(path):
    records = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[(item["candidate_number"], item["seed"])] = item
    return records


def append_record(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def update_tables(output, records, plan, selected=None):
    rows = sorted(records.values(), key=lambda r: (r["candidate_number"], r["seed"]))
    with (output / "trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_number", "stage", "seed", "status",
                                                   "val_mse", "val_mean_error_3d", "val_p95_error_3d",
                                                   "best_epoch", "run_dir", "parameters", "configuration"])
        writer.writeheader()
        for record in rows:
            writer.writerow({**{key: record.get(key) for key in writer.fieldnames},
                             "configuration": json.dumps(plan["candidates"][record["candidate_number"] - 1],
                                                         ensure_ascii=False, sort_keys=True)})
    if selected is None:
        return
    ranking = []
    for number in selected:
        values = [records.get((number, seed)) for seed in plan["confirm_seeds"]]
        if not all(item is not None and item["status"] == "ok" for item in values):
            continue
        scores = [item["val_mse"] for item in values]
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
        ranking.append(dict(candidate_number=number, confirm_mean_mse=mean,
                            confirm_std_mse=std, confirm_values=scores,
                            configuration=plan["candidates"][number - 1]))
    ranking.sort(key=lambda item: item["confirm_mean_mse"])
    save_json(output / "confirmed_ranking.json", ranking)
    with (output / "confirmed_ranking.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "candidate_number", "confirm_mean_mse", "confirm_std_mse",
                         "seed_mse", "configuration"])
        for position, item in enumerate(ranking, 1):
            writer.writerow([position, item["candidate_number"], item["confirm_mean_mse"],
                             item["confirm_std_mse"], json.dumps(item["confirm_values"]),
                             json.dumps(item["configuration"], sort_keys=True)])


def execute_trial(number, seed, stage, plan, output):
    base = output / ("candidate_{:03d}_seed_{}".format(number, seed))
    run_dir = base
    attempt = 1
    while run_dir.exists():
        run_dir = output / (base.name + "_retry_{}".format(attempt))
        attempt += 1
    candidate = plan["candidates"][number - 1]
    command = [plan["python"], plan["runner"], "--data", plan["data"],
               "--output", str(run_dir), "--skip-test", "--mode", "train",
               "--device", plan["device"], "--epochs", str(plan["epochs"]),
               "--patience", str(plan["patience"]), "--seed", str(seed)]
    for key, value in candidate.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    print("{}/{} | {} | seed {} | {}".format(number, len(plan["candidates"]),
                                            stage, seed, run_dir.name), flush=True)
    with (output / (run_dir.name + ".stdout.log")).open("w", encoding="utf-8") as stdout, \
         (output / (run_dir.name + ".stderr.log")).open("w", encoding="utf-8") as stderr:
        finished = subprocess.run(command, cwd=str(HERE), stdout=stdout, stderr=stderr,
                                  check=False)
    metric_path = run_dir / "selection_metrics.json"
    record = dict(candidate_number=number, stage=stage, seed=seed, run_dir=str(run_dir),
                  status="failed", return_code=finished.returncode)
    if finished.returncode == 0 and metric_path.is_file():
        try:
            with metric_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            validation = metrics["validation"]
            record.update(status="ok", val_mse=float(validation["coordinate_mse"]),
                          val_mean_error_3d=float(validation["mean_error_3d"]),
                          val_p95_error_3d=float(validation["p95_error_3d"]),
                          best_epoch=int(metrics["best_epoch"]),
                          parameters=int(metrics["parameters"]))
        except (OSError, KeyError, ValueError, TypeError) as error:
            record["error"] = str(error)
    else:
        record["error"] = "See stderr log; no valid validation metric"
    print("  {} | val MSE {}".format(record["status"], record.get("val_mse")), flush=True)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "prepared.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue an existing overnight-search directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", type=int, default=72,
                        help="Total screening candidates, including the current baseline")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Screening winners to verify with new seeds")
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--confirm-seeds", type=int, nargs="+", default=[2026, 3407, 9091])
    parser.add_argument("--sample-seed", type=int, default=20260926)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true", help="Save a plan without training")
    args = parser.parse_args(argv)
    if args.trials < 2 or args.top_k < 1 or args.top_k > args.trials or args.epochs < 1 or args.patience < 0:
        parser.error("Invalid search size, epoch count or patience")
    if len(set(args.confirm_seeds + [args.screen_seed])) != len(args.confirm_seeds) + 1:
        parser.error("Screen and confirmation seeds must all be different")
    if args.output and args.resume:
        parser.error("Use --output for a new search or --resume for an existing one")
    if args.resume and args.dry_run:
        parser.error("--dry-run is only for a new plan")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.resume:
        output = args.resume.resolve()
        with (output / "plan.json").open(encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("architecture") != ARCHITECTURE or plan["data_sha256"] != digest(plan["data"]) or plan["run_sha256"] != digest(RUNNER) \
                or plan["model_sha256"] != digest(HERE / "model.py"):
            raise RuntimeError("Data or model/runner code changed; use the original files to resume")
    else:
        output = (args.output or HERE / "searches" /
                  ("overnight_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
        if output.exists():
            raise FileExistsError("Choose a new output directory: " + str(output))
        plan = dict(architecture=ARCHITECTURE, runner=str(RUNNER.resolve()), python=sys.executable, data=str(args.data.resolve()),
                    data_sha256=digest(args.data), run_sha256=digest(RUNNER),
                    model_sha256=digest(HERE / "model.py"), device=args.device,
                    epochs=args.epochs, patience=args.patience, screen_seed=args.screen_seed,
                    confirm_seeds=args.confirm_seeds, top_k=args.top_k,
                    candidates=make_candidates(args.trials, args.sample_seed))
        output.mkdir(parents=True)
        save_json(output / "plan.json", plan)
    print("Search: {} | {} screening + up to {} confirmation runs".format(
        output, len(plan["candidates"]), plan["top_k"] * len(plan["confirm_seeds"])), flush=True)
    if args.dry_run:
        print("Plan only; no training started.", flush=True)
        return
    records_path = output / "records.jsonl"
    records = read_records(records_path)
    for number in range(1, len(plan["candidates"]) + 1):
        key = (number, plan["screen_seed"])
        if key not in records:
            record = execute_trial(number, plan["screen_seed"], "screen", plan, output)
            append_record(records_path, record)
            records[key] = record
            update_tables(output, records, plan)
    successful = [records[(number, plan["screen_seed"])]
                  for number in range(1, len(plan["candidates"]) + 1)
                  if records[(number, plan["screen_seed"])]["status"] == "ok"]
    if not successful:
        raise RuntimeError("All screening runs failed; inspect stderr logs")
    selected = [record["candidate_number"] for record in
                sorted(successful, key=lambda record: record["val_mse"])[:plan["top_k"]]]
    save_json(output / "selected_for_confirmation.json", selected)
    for number in selected:
        for seed in plan["confirm_seeds"]:
            key = (number, seed)
            if key not in records:
                record = execute_trial(number, seed, "confirm", plan, output)
                append_record(records_path, record)
                records[key] = record
                update_tables(output, records, plan, selected)
    update_tables(output, records, plan, selected)
    with (output / "confirmed_ranking.json").open(encoding="utf-8") as handle:
        ranking = json.load(handle)
    if not ranking:
        raise RuntimeError("No candidate passed every confirmation seed; inspect stderr logs")
    best = ranking[0]
    print("Best confirmed candidate {} | mean val MSE {:.6f} +/- {:.6f}".format(
        best["candidate_number"], best["confirm_mean_mse"], best["confirm_std_mse"]), flush=True)
    print("Configuration: " + json.dumps(best["configuration"], ensure_ascii=False), flush=True)
    print("Ranking: " + str(output / "confirmed_ranking.csv"), flush=True)


if __name__ == "__main__":
    main()
