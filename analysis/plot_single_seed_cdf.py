"""沿用所提供多种子脚本的模型、样式和坐标筛选，绘制单种子经验CDF。

只评估保存的best.pt；图例显示单次平均3D欧几里得误差，不计算种子标准差。
兼容runs目录下seed42_时间戳和时间戳两种布局。--all-test可显式评估完整测试集。
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = (
    ("sdfnet_full", "SDF-Net"),
    # ("ablation_ca_only", "SDF-Net (CA only)"),
    ("ablation_no_ca_mhsa", "SDF-Net (w/o MHSA)"),
    ("sdfnet_full_all14", "All 11 stats + SAGE"),
    ("ablation_sage_only", "SAGE only"),
    ("ablation_stat_only", "Statistics only"),
    ("single_stream_sdfnet", "Single-stream SDF-Net"),
    ("open_cnn_transformer", "ReConFormer (adapted)"),
    ("open_cnn", "CNN (adapted)"),
    ("open_transformer", "WiT(adapted)"),
    ("sdfnet_rf_rfe", "RF-RFE SDF-Net"),
)

COORDINATE_FILTER_EXPRESSION = '((((x >= 0.5) & (x <= 3.7)) | ((x >= 4.7) & (x <= 6.5)) | ((x >= 7.5) & (x <= 8.4)))& ((y < 2.2) | (y > 7.25))& ~(((x >= 0.5) & (x <= 3.7) & (y < 2.2) & (z > 2.45) & (z < 2.55))| ((x > 4.95) & (x < 5.05) & (y > 0.95) & (y < 1.01) & (z > 2.47) & (z < 2.55))))'

OPEN_MODELS = {
    "open_cnn",
    "open_cnn_transformer",
    "open_transformer",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_directory(project, key, runs_dir, tag, seed):
    """优先避免目录歧义；新旧布局同时存在时不静默选择。"""
    root = project / "experiments" / key / runs_dir
    candidates = [root / f"seed{seed}_{tag}", root / tag]
    found = []
    for path in candidates:
        if path.is_dir() and path not in found:
            found.append(path)
    if len(found) != 1:
        raise ValueError(f"应恰好找到一个seed{seed}运行目录，实际为{found}；模型={key}，批次={tag}")
    return found[0]


def check_batch(project, runs_dir, tag, seeds, data_hash, expected_splits):
    """检查每个模型的指定种子已完成，并核对源码、数据与划分。"""
    runs = {}
    for key, _ in MODELS:
        reference_settings = None
        for seed in seeds:
            run = run_directory(project, key, runs_dir, tag, seed)
            if key in OPEN_MODELS:
                for name in (
                    "best.pt", "config.json", "validation_metrics.json",
                    "run.py", "model.py",
                ):
                    if not (run / name).is_file():
                        raise FileNotFoundError(run / name)

                config = read_json(run / "config.json")
                arguments = config["arguments"]

                if arguments.get("mode") != "train" or arguments.get("seed") != seed:
                    raise ValueError("运行模式或种子不匹配：" + str(run))

                metrics = read_json(run / "validation_metrics.json")
                if not metrics.get("best_epoch"):
                    raise ValueError("训练尚未完成：" + str(run))

                if config.get("source_sha256") != data_hash:
                    raise ValueError("训练数据与当前NPZ不同：" + str(run))

                metadata = (
                    Path(arguments["metadata"])
                    if arguments.get("metadata")
                    else Path(arguments["data"]).with_suffix(".json")
                )
                if sha256(metadata) != config.get("metadata_sha256"):
                    raise ValueError("配套JSON与训练记录不同：" + str(run))

                hashes = config.get("source_hashes", {})
                if not {"run.py", "model.py"}.issubset(hashes):
                    raise ValueError("缺少源码哈希：" + str(run))

                for name, digest in hashes.items():
                    if sha256(run / name) != digest:
                        raise ValueError("保存源码与训练记录不同：" + str(run / name))

                settings = {
                    "arguments": {
                        k: v for k, v in arguments.items()
                        if k not in ("seed", "output")
                    },
                    **{
                        k: config.get(k) for k in (
                            "packing_version", "statistical_features",
                            "physical_parameters", "model_config",
                            "source_hashes", "versions", "validation_policy",
                            "source_sha256", "metadata_sha256",
                        )
                    },
                }

                if reference_settings is None:
                    reference_settings = settings
                elif settings != reference_settings:
                    raise ValueError("同模型不同种子的配置或源码不同：" + str(run))

                runs[(key, seed)] = run
                continue
            for name in ("best.pt", "config.json", "selection_metrics.json", "run.py", "model.py"):
                if not (run / name).is_file():
                    raise FileNotFoundError(run / name)
            config = read_json(run / "config.json")
            if config.get("mode") != "train" or config.get("seed") != seed:
                raise ValueError("运行模式或种子不匹配：" + str(run))
            if not config.get("best_epoch") or "training_seconds" not in config:
                raise ValueError("训练尚未完成：" + str(run))
            if config.get("data_sha256") != data_hash:
                raise ValueError("训练数据与当前NPZ不同：" + str(run))
            for name in ("run.py", "model.py"):
                if config.get("source_hashes", {}).get(name) != sha256(run / name):
                    raise ValueError("保存源码与训练记录不同：" + str(run / name))
            splits = config.get("split_ids")
            if not splits or set(splits) != {"train", "val"}:
                raise ValueError("缺少训练/验证样本编号：" + str(run))
            for name in ("train", "val"):
                ids = splits[name]
                if len(ids) != len(expected_splits[name]) or len(set(ids)) != len(ids) or set(ids) != set(expected_splits[name]):
                    raise ValueError("训练/验证划分与NPZ不一致：" + str(run))
            # 除种子、输出路径、训练耗时及最佳轮数外，同模型三次设置应相同。
            settings = {k: config.get(k) for k in (
                "architecture", "model_config", "input_config", "statistical_features",
                "epochs", "batch_size", "lr", "weight_decay", "patience", "max_paths",
                "angle_encoding", "optimizer", "scheduler", "warmup_epochs", "min_lr_ratio",
                "ema_decay", "grad_clip", "aux_weight", "distance_weight", "feature_noise",
                "selection_metric", "source_hashes", "torch_version", "numpy_version")}
            if reference_settings is None:
                reference_settings = settings
            elif settings != reference_settings:
                raise ValueError("同模型不同种子的配置或源码不同：" + str(run))
            runs[(key, seed)] = run
    return runs


def select_batch(project, runs_dir, requested, seeds, data_hash, expected_splits):
    if requested:
        return requested, check_batch(project, runs_dir, requested, seeds, data_hash, expected_splits)
    candidates = None
    for key, _ in MODELS:
        root = project / "experiments" / key / runs_dir
        if not root.is_dir():
            raise FileNotFoundError(root)
        for seed in seeds:
            prefix = f"seed{seed}_"
            tags = set()
            for path in root.iterdir():
                if not path.is_dir() or not (path / "config.json").is_file():
                    continue
                config = read_json(path / "config.json")
                arguments = config.get("arguments", config)
                if arguments.get("mode") != "train" or arguments.get("seed") != seed:
                    continue
                tags.add(path.name[len(prefix):] if path.name.startswith(prefix) else path.name)
            candidates = tags if candidates is None else candidates & tags
    failures = []
    for tag in sorted(candidates or (), reverse=True):
        try:
            return tag, check_batch(project, runs_dir, tag, seeds, data_hash, expected_splits)
        except (ValueError, FileNotFoundError) as exc:
            failures.append(f"{tag}: {exc}")
    raise RuntimeError("没有找到覆盖模型列表、数据一致且训练完成的单种子批次。" + " | ".join(failures[:3]))


def summarize(errors, seeds):
    rows = []
    for key, label in MODELS:
        values = errors[key]  # [种子数, 测试样本数]，所有样本参与评估
        seed_means = values.mean(axis=1)
        rows.append({"model": label, "key": key, "n_per_seed": values.shape[1],
                     "seeds": ",".join(map(str, seeds)),
                     "mean_error_3d": float(seed_means.mean()),
                     "seed": int(seeds[0]),
                     "P50": float(np.median(values[0])),
                     "P90": float(np.percentile(values[0], 90))})
    rows.sort(key=lambda r: r["mean_error_3d"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def draw(errors, rows, output):
    """保留原配色、线型、字体和图尺寸，直接绘制单次经验CDF。"""
    fig, ax = plt.subplots(figsize=(10.6, 7.4))
    palette = plt.get_cmap("tab20")
    colors = {key: palette(i) for i, (key, _) in enumerate(MODELS)}
    for row in rows:
        key = row["key"]
        values = errors[key]
        grid = np.unique(np.concatenate((np.array([0.0]), values.ravel())))
        curve = np.searchsorted(np.sort(values[0]), grid, side="right") / len(values[0])
        label = f"{row['model']} ({row['mean_error_3d']:.3f} m)"
        ax.step(grid, curve, where="post", color=colors[key], label=label,
                linewidth=2.5 if key == "sdfnet_full" else 1.6,
                linestyle="--" if key == "sdfnet_rf_rfe" else "-")
    ax.set(xlim=(0, None), ylim=(0, 1.01))

    ax.set_xlabel("Error [m]", fontname="Times New Roman", fontsize=24)
    ax.set_ylabel(
        "P(Error [m] ≤ abscissa)",
        fontname="Times New Roman",
        fontsize=24,
    )

    ax.tick_params(axis="both", labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    ax.grid(alpha=.22)
    ax.legend(
        loc="lower right",
        prop={"family": "Times New Roman", "size": 10.2},
        framealpha=.95,
    )
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / ("cdf." + extension), dpi=240)
    plt.close(fig)


def infer_open(run, data, device, output):
    config = read_json(run / "config.json")
    arguments = config["arguments"]

    metadata = (
        Path(arguments["metadata"])
        if arguments.get("metadata")
        else data.with_suffix(".json")
    )

    command = [
        sys.executable, str(run / "run.py"),
        "--mode", "test",
        "--checkpoint", str(run / "best.pt"),
        "--data", str(data),
        "--metadata", str(metadata),
        "--device", device,
        "--output", str(output),
        "--batch-size", str(arguments["batch_size"]),
        "--seed", str(arguments["seed"]),
    ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "inference_stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output / "inference_stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )

    if completed.returncode:
        raise RuntimeError(
            "测试失败，请查看：" + str(output / "inference_stderr.txt")
        )

    with (output / "test_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        records = list(csv.DictReader(stream))

    ids = [r["array_id"] for r in records]
    truth = np.asarray([
        [float(r["true_" + a]) for a in "xyz"] for r in records
    ])
    pred = np.asarray([
        [float(r["pred_" + a]) for a in "xyz"] for r in records
    ])

    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("预测或坐标含非有限数值")

    return ids, truth, pred


def device_name(requested):
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请选择--device cpu")
    return requested


def aligned_errors(ids, truth, pred, reference):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("测试array_id为空或重复")
    if truth.shape != pred.shape or truth.shape != (len(ids), 3):
        raise ValueError("测试坐标/预测形状不一致")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("测试坐标或预测含非有限值")
    order = np.argsort(np.asarray(ids, dtype=str))
    ids, truth, pred = [ids[i] for i in order], truth[order], pred[order]
    if ids != reference[0] or not np.allclose(truth, reference[1], rtol=0, atol=1e-8):
        raise ValueError("预测结果未覆盖同一完整测试集，或坐标不一致")
    return ids, truth, np.linalg.norm(pred-truth, axis=1)


def infer_other(project, run, checkpoint, data, device, output):
    config = read_json(run / "config.json")
    command = [sys.executable, "-B", str(run / "run.py"), "--mode", "test",
               "--checkpoint", str(checkpoint), "--data", str(data),
               "--device", device, "--output", str(output),
               "--batch-size", str(config["batch_size"]), "--seed", str(config["seed"])]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inference_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "inference_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("测试失败，请查看：" + str(output / "inference_stderr.txt"))
    files = [p for p in (output / "test_predictions.csv", output / "predictions.csv") if p.is_file()]
    if len(files) != 1:
        raise ValueError("缺少唯一的测试预测CSV：" + str(output))
    with files[0].open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    ids = [r["array_id"] for r in records]
    truth = np.asarray([[float(r["true_"+a]) for a in "xyz"] for r in records], dtype=np.float64)
    pred = np.asarray([[float(r["pred_"+a]) for a in "xyz"] for r in records], dtype=np.float64)
    return ids, truth, pred


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--runs-dir", default="runs_latest_3seeds")
    parser.add_argument("--tag", help="整批时间戳，例如20261019_040000；省略时选最新完整批次")
    parser.add_argument("--seed", type=int, default=42, help="只绘制此种子，默认42")
    parser.add_argument("--all-test", action="store_true", help="显式关闭原坐标筛选，使用全部测试位置")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="可选的全新输出目录")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed必须为0到2**32-1之间的整数")
    seeds = [args.seed]
    if args.runs_dir in ("", ".", "..") or Path(args.runs_dir).name != args.runs_dir:
        parser.error("--runs-dir必须是实验目录下面的一个文件夹名称")
    project = args.project.resolve()
    data = (args.data or project / "data/prepared.npz").resolve()
    data_hash = sha256(data)
    with np.load(data, allow_pickle=False) as archive:
        expected_splits = {name: archive["array_id"][archive[name + "_indices"]].tolist() for name in ("train", "val")}
        test_ids = [str(v) for v in archive["array_id"][archive["test_indices"]]]
        test_truth = archive["target"][archive["test_indices"]].astype(np.float64)
    if len(test_ids) != len(set(test_ids)) or not len(test_ids):
        raise ValueError("NPZ测试编号为空或重复")
    order = np.argsort(np.asarray(test_ids, dtype=str))
    reference = [test_ids[i] for i in order], test_truth[order]
    # 按排序后的真实坐标判断 NLOS，与后面的误差顺序保持一致
    truth_sorted = reference[1]
    x = truth_sorted[:, 0]
    y = truth_sorted[:, 1]
    z = truth_sorted[:, 2]
    # nlos的点的判断 边缘区域
    nlos_mask = ((((x >= 0.5) & (x <= 3.7)) | ((x >= 4.7) & (x <= 6.5)) | ((x >= 7.5) & (x <= 8.4)))& ((y < 2.2) | (y > 7.25))& ~(((x >= 0.5) & (x <= 3.7) & (y < 2.2) & (z > 2.45) & (z < 2.55))| ((x > 4.95) & (x < 5.05) & (y > 0.95) & (y < 1.01) & (z > 2.47) & (z < 2.55))))
    # 保持粘贴脚本的坐标条件原样；该条件不读取任何模型预测误差。
    if args.all_test:
        nlos_mask = np.ones(len(reference[0]), dtype=bool)
    n_nlos = int(nlos_mask.sum())
    if n_nlos == 0:
        raise ValueError("原坐标条件未选中任何测试点；请核对数据或显式使用--all-test")
    print(f"全部测试点：{len(test_ids)} | 本次评估点：{n_nlos} | seed={args.seed}", flush=True)
    tag, runs = select_batch(project, args.runs_dir, args.tag, seeds, data_hash, expected_splits)
    print(f"Batch: {tag} | runs: {len(runs)} | test points per seed: {len(test_ids)}", flush=True)
    if args.dry_run:
        print("Dry run passed; no inference or output files.", flush=True)
        return
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "analysis"))
    import torch
    # 测试推理及对齐函数包含在本脚本中，不依赖另一份旧绘图脚本。
    torch.set_num_threads(2)
    device = device_name(args.device)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or project / "analysis/cdf_runs" / f"single_seed{args.seed}_{tag}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"batch": tag, "data": str(data), "data_sha256": data_hash,
                  "seed": args.seed, "device": device, "evaluation_subset": "all test positions" if args.all_test else "original coordinate-defined region",
                  "source_filter_note": "原粘贴脚本的坐标条件；未在此脚本中独立验证LoS/NLoS标签",
"n_total_test": len(test_ids),
"n_per_seed": n_nlos,
"evaluated_array_ids": [
    reference[0][i] for i in np.flatnonzero(nlos_mask)
],
                  "runs_dir": args.runs_dir, "std_definition": "not applicable: one seed; no between-seed standard deviation",
                  "cdf_definition": "single-seed empirical CDF, all selected positions included",
                  "runs": {}}
    errors, per_seed = {}, []
    for key, label in MODELS:
        seed_errors = []
        for seed in seeds:
            run = runs[(key, seed)]
            destination = output / "predictions" / f"{key}_seed{seed}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if key in OPEN_MODELS:
                ids, truth, pred = infer_open(
                    run, data, device, destination
                )
            else:
                ids, truth, pred = infer_other(
                    project, run, run / "best.pt", data, device, destination
                )
            # 先核对并对齐完整测试集，再保留 NLOS 点的误差
            _, _, distance = aligned_errors(ids, truth, pred, reference)
            distance = distance[nlos_mask]

            seed_errors.append(distance)
            per_seed.append({"model": label, "seed": seed, "n": len(distance),
                             "mean_error_3d": float(distance.mean())})
            provenance["runs"][f"{key}/seed{seed}"] = {"path": str(run), "checkpoint_sha256": sha256(run / "best.pt")}
            print(f"{label} seed{seed}: {distance.mean():.6f} m", flush=True)
        errors[key] = np.stack(seed_errors)
    rows = summarize(errors, seeds)
    provenance["plot_script_sha256"] = sha256(Path(__file__).resolve())
    provenance["coordinate_filter_expression"] = COORDINATE_FILTER_EXPRESSION
    draw(errors, rows, output)
    for name, records in (("summary.csv", rows), ("per_seed.csv", per_seed)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / "selected_runs.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / "test_errors.npz", **errors)
    for row in rows:
        print(f"{row['rank']:2d}. {row['model']}: {row['mean_error_3d']:.6f} m", flush=True)
    print("CDF saved to:", output, flush=True)


if __name__ == "__main__":
    main()
