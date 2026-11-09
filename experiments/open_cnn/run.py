"""适配作者纯CNN骨干的独立实验；固定NPZ划分，训练仅用训练/验证集。

用户确认：使用prepared.npz；迁移PyTorch；保留全部路径；补零后拼接单流。
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import OpenCNN


ROOT = Path(__file__).resolve().parent
SELECTED_STAT_COLUMNS = [
    "Tau_MED_ns", "K_factor", "PL_dB", "Rx_power_dBm",
    "tau_dom_RMS_ns", "DPR", "R_max_dB",
]
PHYSICAL_COLUMNS = ["delay_ns", "AoA_deg", "EoA_deg"]
MAX_PATHS = 15
UPSTREAM = "https://github.com/sibrendebast/MaMIMO-CSI-positioning-using-CNNs/blob/master/positioning.py"
PACKING_VERSION = "stat16_path15_circular5_pad7_single_stream_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_inputs(source, metadata_path, wanted):
    """按用户确认的排列构造单流；只提取本次需要的划分。"""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    names = metadata["statistical_columns"]
    physical_names = metadata["physical_columns"]
    if len(names) != 11 or len(set(names)) != 11 or set(physical_names) != set(PHYSICAL_COLUMNS):
        raise ValueError("JSON中的候选特征定义不正确")
    stat_indices = [names.index(name) for name in SELECTED_STAT_COLUMNS]
    phys_order = [physical_names.index(name) for name in PHYSICAL_COLUMNS]
    result = {}
    with np.load(source, allow_pickle=False) as archive:
        array_ids = archive["array_id"].astype(str)
        total = len(array_ids)
        if array_ids.ndim != 1 or len(set(array_ids)) != total:
            raise ValueError("阵列ID必须唯一")
        splits = {name: archive[name + "_indices"] for name in ("train", "val", "test")}
        if any(idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer)
               for idx in splits.values()):
            raise ValueError("划分索引必须为非空整数数组")
        if not np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(total)):
            raise ValueError("训练/验证/测试划分有重叠、遗漏或越界")

        for key, count in (("stat_mean", 11), ("stat_scale", 11),
                           ("phys_mean", 3), ("phys_scale", 3)):
            value = np.asarray(archive[key], dtype=np.float64)
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError("非法标准化参数：" + key)
            if "scale" in key and np.any(value <= 0):
                raise ValueError("标准化尺度必须为正")
            if not np.array_equal(np.asarray(metadata["statistics"][key]), value):
                raise ValueError("NPZ和JSON标准化记录不一致：" + key)

        mean, scale = archive["phys_mean"], archive["phys_scale"]
        for split_name in wanted:
            idx = splits[split_name]
            stats = archive["statistical"][idx].astype(np.float32)
            phys = archive["physical"][idx].astype(np.float64)
            mask = archive["path_mask"][idx]
            targets = archive["target"][idx].astype(np.float32)
            n = len(idx)
            if (stats.shape != (n, 16, 11) or phys.shape != (n, MAX_PATHS, 3)
                    or mask.shape != (n, MAX_PATHS) or mask.dtype != np.bool_
                    or targets.shape != (n, 3)):
                raise ValueError("NPZ输入形状不符合当前实验定义")
            if not np.isfinite(stats).all() or not np.isfinite(phys[mask]).all() or not np.isfinite(targets).all():
                raise ValueError("输入含非有限值，请核对数据准备过程")

            # 统计流：16根天线，每根7项统计量；直接使用NPZ的训练集标准化结果。
            stats = stats[:, :, stat_indices]
            # 保留全部有效路径，按时延升序排列；不筛掉低功率路径。
            ordered = np.zeros_like(phys)
            for row in range(n):
                valid = np.flatnonzero(mask[row])
                delay_column = physical_names.index("delay_ns")
                valid = valid[np.argsort(phys[row, valid, delay_column], kind="stable")]
                ordered[row, :len(valid)] = phys[row, valid]
            valid_counts = mask.sum(axis=1)
            ordered_mask = np.arange(MAX_PATHS)[None, :] < valid_counts[:, None]
            raw = ordered * scale + mean
            raw = raw[:, :, phys_order]
            ordered = ordered[:, :, phys_order]
            # 与当前SDF-Net一致，把两个角度变为sin/cos，保留标准化时延。
            # 仍是3类物理参数，仅展开成5个数值通道，不新增信道特征类别。
            aoa, eoa = np.deg2rad(raw[:, :, 1]), np.deg2rad(raw[:, :, 2])
            path_features = np.stack((ordered[:, :, 0], np.sin(aoa), np.cos(aoa),
                                      np.sin(eoa), np.cos(eoa)), axis=-1)
            path_features[~ordered_mask] = 0

            # 用户确认的单流排列：前16行为统计天线，后15行为全部物理路径。
            # 物理行只有5个数值通道，尾部补2个零；无效路径整行为零。
            # 不把这些列当作复数实部/虚部，也不在这些指纹上执行IFFT。
            packed = np.zeros((n, 1, 16 + MAX_PATHS, 7, 1), dtype=np.float32)
            packed[:, 0, :16, :, 0] = stats
            packed[:, 0, 16:, :5, 0] = path_features
            result[split_name] = dict(inputs=packed, target=targets,
                                      array_id=array_ids[idx], valid_path_counts=valid_counts)
    return result


def metrics(prediction, target):
    errors = np.linalg.norm(prediction.astype(np.float64) - target, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("预测误差非有限")
    return dict(n=len(errors), coordinate_mse=float(np.mean((prediction.astype(np.float64) - target) ** 2)),
                mean_error_3d=float(errors.mean()), median_error_3d=float(np.percentile(errors, 50)),
                p90_error_3d=float(np.percentile(errors, 90)))


def evaluate(model, split, device, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(split["target"]), batch_size):
            inputs = torch.from_numpy(split["inputs"][start:start + batch_size]).to(device)
            predictions.append(model(inputs).cpu().numpy())
    prediction = np.concatenate(predictions)
    return prediction, metrics(prediction, split["target"])


def export_predictions(path, split, prediction):
    errors = np.linalg.norm(prediction.astype(np.float64) - split["target"], axis=1)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["array_id", "true_x", "true_y", "true_z",
                         "pred_x", "pred_y", "pred_z", "error_3d"])
        for array_id, target, pred, error in zip(split["array_id"], split["target"], prediction, errors):
            writer.writerow([array_id, *target.tolist(), *pred.tolist(), float(error)])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--source", dest="data", type=Path,
                        default=Path("/data/prepared.npz"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--mode", choices=("check", "train", "test"), default="check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("训练参数必须为正且有限")
    if args.mode == "test" and args.checkpoint is None:
        parser.error("测试必须显式传入 --checkpoint")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用CUDA")
    device = torch.device(device_name)
    # 小CNN在CPU上使用少量线程，避免大量小卷积的线程调度开销。
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    metadata_path = args.metadata or args.data.with_suffix(".json")
    wanted = ("test",) if args.mode == "test" else (("train", "val") if args.mode == "train" else ("train",))
    print("读取固定划分：" + ", ".join(wanted), flush=True)
    data = load_inputs(args.data, metadata_path, wanted)
    model_config = dict(input_rows=31, input_columns=7, output_dim=3)
    model = OpenCNN(**model_config).to(device)
    print("输入 [B,1,31,7,1]；全部15个路径槽；输出xyz；设备：" + str(device), flush=True)
    print("参数数量：{}".format(sum(value.numel() for value in model.parameters())), flush=True)

    if args.mode == "check":
        # 仅用少量训练样本检查前向、反向和梯度，不训练或读取测试特征。
        batch = torch.from_numpy(data["train"]["inputs"][:4]).to(device)
        target = torch.from_numpy(data["train"]["target"][:4]).to(device)
        prediction = model(batch)
        loss = (prediction - target).square().mean() + model.kernel_regularization()
        loss.backward()
        if prediction.shape != (len(batch), 3) or not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise ValueError("前向结果异常")
        if any(value.grad is None or not torch.isfinite(value.grad).all() for value in model.parameters()):
            raise ValueError("梯度缺失或非有限")
        print("检查通过：真实训练样本前向/反向、三维输出和所有梯度均有效。", flush=True)
        return

    output = args.output or ROOT / "runs" / ("seed{}_".format(args.seed) + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in ("model.py", "run.py", "upstream_model_reference.py"):
        source = ROOT / name
        if source.exists():
            source_hashes[name] = sha256(source)
            shutil.copy2(source, output / name)
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=sha256(args.data), metadata_sha256=sha256(metadata_path),
                  packing_version=PACKING_VERSION, statistical_features=SELECTED_STAT_COLUMNS,
                  physical_parameters=PHYSICAL_COLUMNS,
                  physical_input_columns=["delay_z", "sin_AoA", "cos_AoA", "sin_EoA", "cos_EoA"],
                  model_config=model_config, source_hashes=source_hashes, upstream=UPSTREAM,
                  validation_policy="early stopping: validation MSE + kernel L2; checkpoint: validation mean 3D error",
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__))
    save_json(output / "config.json", config)

    if args.mode == "test":
        # 只加载自己产生且来源可信的检查点；数据和排列方式必须与训练一致。
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["source_sha256"] != config["source_sha256"] or saved["metadata_sha256"] != config["metadata_sha256"]:
            raise ValueError("测试数据或元数据与训练记录不一致")
        if saved["packing_version"] != PACKING_VERSION or saved["model_config"] != model_config:
            raise ValueError("检查点输入排列或网络配置不一致")
        model.load_state_dict(saved["model"])
        prediction, report = evaluate(model, data["test"], device, args.batch_size)
        export_predictions(output / "test_predictions.csv", data["test"], prediction)
        save_json(output / "test_metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("结果目录：" + str(output), flush=True)
        return

    # 原作者Adam默认学习率0.001；Keras默认epsilon为1e-7。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(data["train"]["inputs"]),
                                     torch.from_numpy(data["train"]["target"])),
                        batch_size=args.batch_size, shuffle=True, generator=generator,
                        drop_last=False, num_workers=0)
    best_error, best_early_loss, stale = float("inf"), float("inf"), 0
    with (output / "history.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_mse", "val_loss", "val_mean_error_3d"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            accumulated, count = 0.0, 0
            for inputs, target in loader:
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(inputs) - target).square().mean() + model.kernel_regularization()
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限")
                loss.backward()
                optimizer.step()
                accumulated += float(loss.detach()) * len(target)
                count += len(target)
            _, report = evaluate(model, data["val"], device, args.batch_size)
            val_loss = report["coordinate_mse"] + float(model.kernel_regularization().detach())
            writer.writerow([epoch, accumulated / count, report["coordinate_mse"], val_loss, report["mean_error_3d"]])
            stream.flush()
            print("epoch {:3d} train={:.6f} val_loss={:.6f} val_mean={:.6f} m".format(
                epoch, accumulated / count, val_loss, report["mean_error_3d"]), flush=True)
            if report["mean_error_3d"] < best_error:
                best_error = report["mean_error_3d"]
                torch.save(dict(model=model.state_dict(), model_config=model_config,
                                source_sha256=config["source_sha256"], metadata_sha256=config["metadata_sha256"],
                                packing_version=PACKING_VERSION, epoch=epoch, seed=args.seed,
                                validation=report), output / "best.pt")
            if val_loss < best_early_loss:
                best_early_loss, stale = val_loss, 0
            else:
                stale += 1
            if stale >= args.patience:
                print("验证损失连续{}轮未改善，停止。".format(args.patience), flush=True)
                break
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    prediction, report = evaluate(model, data["val"], device, args.batch_size)
    export_predictions(output / "validation_predictions.csv", data["val"], prediction)
    save_json(output / "validation_metrics.json", dict(best_epoch=saved["epoch"], **report))
    print("训练完成；测试需显式运行 --mode test。结果目录：" + str(output), flush=True)


if __name__ == "__main__":
    main()
