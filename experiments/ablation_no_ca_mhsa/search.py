"""Validation-only SDF-Net search, reusing common/hyperparameter_search.py."""
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))
from run import PROJECT, build_parser, validate_args


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--stage", choices=("training", "architecture", "regularization"), default="training")
    parser.add_argument("--dry-run", action="store_true", help="Write commands without training")
    parser.add_argument("--seeds", type=int, nargs="+", help="Repeat each candidate; rank configuration mean, not best seed")
    args = validate_args(parser.parse_args(argv), parser)
    if args.mode != "train" or args.checkpoint:
        parser.error("Search only supports training from scratch")
    grids = {
        "training": dict(lr=[1e-3, 3e-4, 1e-4], weight_decay=[0.0, 1e-4, 1e-3]),
        "architecture": dict(d_model=[64, 96, 128], cnn_width=[24, 32, 48]),
        "regularization": dict(dropout=[0.0, 0.1, 0.2], ema_decay=[0.0, 0.98, 0.995]),
    }
    grid = grids[args.stage]
    if args.stage == "architecture" and any(d % args.num_heads for d in grid["d_model"]):
        parser.error("Every candidate d-model must be divisible by num-heads")
    if args.seeds:
        grid["seed"] = list(dict.fromkeys(args.seeds))
    excluded = {"stage", "dry_run", "seeds", "mode", "checkpoint", "output", "skip_test"} | set(grid)
    base = {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key not in excluded}
    output = args.output or EXPERIMENT / "searches" / (args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    sys.path.insert(0, str(PROJECT / "common"))
    from hyperparameter_search import run_grid
    records = run_grid(EXPERIMENT / "run.py", output, grid, base,
                       metric_key="validation." + args.selection_metric,
                       python_executable=sys.executable, dry_run=args.dry_run)
    if args.dry_run:
        print("Planned {} trials, no training. {}".format(len(records), output))
        return
    grouped = {}
    for record in records:
        config = {k: v for k, v in record["candidate"].items() if k != "seed"}
        key = json.dumps(config, sort_keys=True)
        group = grouped.setdefault(key, dict(config=config, values=[], trials=0))
        group["trials"] += 1
        if record["status"] == "ok":
            group["values"].append(record["metric"])
    rows = []
    for group in grouped.values():
        values = group["values"]
        complete = len(values) == group["trials"]
        mean = sum(values) / len(values) if values else None
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 if values else None
        rows.append(dict(config=group["config"], complete=complete, successful=len(values),
                         expected=group["trials"], mean=mean, std=std))
    rows.sort(key=lambda r: (not r["complete"], r["mean"] if r["mean"] is not None else float("inf")))
    with (output / "config_ranking.json").open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (output / "config_ranking.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "complete", "successful", "expected", "mean", "std"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, config=json.dumps(row["config"], sort_keys=True)))
    successful = [row for row in rows if row["complete"]]
    if not successful:
        raise SystemExit("No complete configuration; inspect trial logs in " + str(output))
    print("Best validation configuration: {} | {} = {:.6f} +/- {:.6f}".format(
        successful[0]["config"], args.selection_metric, successful[0]["mean"], successful[0]["std"]))


if __name__ == "__main__":
    main()
