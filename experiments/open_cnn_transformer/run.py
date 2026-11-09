
import argparse  # 导入数据读取、训练或结果记录所需组件。
import csv  # 导入数据读取、训练或结果记录所需组件。
import hashlib  # 导入数据读取、训练或结果记录所需组件。
import json  # 导入数据读取、训练或结果记录所需组件。
import random  # 导入数据读取、训练或结果记录所需组件。
import shutil  # 导入数据读取、训练或结果记录所需组件。
import sys  # 导入数据读取、训练或结果记录所需组件。
from datetime import datetime  # 导入数据读取、训练或结果记录所需组件。
from pathlib import Path  # 导入数据读取、训练或结果记录所需组件。

import numpy as np  # 导入数据读取、训练或结果记录所需组件。
import torch  # 导入数据读取、训练或结果记录所需组件。
from torch.utils.data import DataLoader, TensorDataset  # 导入数据读取、训练或结果记录所需组件。

from models import original_model  # 加载仅修复一处语法错误的模型运行副本，保留根目录原始model.py。


class AdaptedReConFormer(torch.nn.Module):  # 在原模型外完成输入和回归任务适配。
    def __init__(self, input_rows=31, input_columns=7, output_dim=3):  # 定义与其他基线一致的指纹信息和三维输出。
        super().__init__()  # 初始化PyTorch模块。
        self.input_rows = input_rows  # 保存16根天线加15个路径槽的行数。
        self.input_columns = input_columns  # 保存补零后的七列特征。
        self.padded_rows = ((input_rows + 3) // 4) * 4  # 补至四的倍数，使原始位置编码长度与降采样输出匹配。
        self.backbone = original_model.R_ConFormer(n_way=output_dim, input_shape=[input_columns, self.padded_rows])  # 原构造器默认128维、8头、CNN循环1次、Transformer循环4次。
        self.backbone.transformer.classifier = torch.nn.Linear(128, output_dim)  # 仅替换距离分类头，输出不受余弦分类分数范围限制的米制xyz。

    def forward(self, values):  # 处理31乘7指纹并调用作者原始forward。
        if tuple(values.shape[1:]) != (1, self.input_rows, self.input_columns):  # 不允许静默改变输入排列。
            raise ValueError("输入必须为[B,1,31,7]")  # 提示正确的单流指纹形状。
        sequence = values[:, 0].transpose(1, 2).contiguous()  # 七列作为Conv1d输入通道，三十一行作为序列。
        sequence = torch.nn.functional.pad(sequence, (0, self.padded_rows - self.input_rows))  # 只在序列末尾补一行零，不插值或重复原特征。
        original_model.device = values.device  # 原代码用模块全局device创建零CLS；外部同步设备以支持显式CPU检查。
        return self.backbone(sequence)  # 保留作者卷积、归一化、池化、循环和CLS取值顺序。


ROOT = Path(__file__).resolve().parent  # 所有默认输出都写到本实验目录。
SELECTED_STAT_COLUMNS = [  # 与open_cnn采用相同的七项统计特征及排列。
    "Tau_MED_ns", "K_factor", "PL_dB", "Rx_power_dBm",  # 保留平均时延、K因子、路径损耗和接收功率。
    "tau_dom_RMS_ns", "DPR", "R_max_dB",  # 保留RMS时延扩展、主径功率比和最大幅度。
]  # 结束统计特征定义。
PHYSICAL_COLUMNS = ["delay_ns", "AoA_deg", "EoA_deg"]  # 三类路径参数的规范顺序。
MAX_PATHS = 15  # 保留NPZ中的全部十五个路径槽。
UPSTREAM = "https://github.com/infinite0522/Recurrent-ConFormer-for-WiFi-HAR/blob/32f93c4df010454de29497352c2d2ca0d7429193/models/RConFormer.py"  # 完成当前步骤的参数设置、计算或结果记录。
PACKING_VERSION = "stat16_path15_circular5_pad7_sequence32_reconformer_v1"  # 完成当前步骤的参数设置、计算或结果记录。


def sha256(path):  # 定义本实验的数据处理或运行函数。
    digest = hashlib.sha256()  # 完成当前步骤的参数设置、计算或结果记录。
    with Path(path).open("rb") as stream:  # 完成当前步骤的参数设置、计算或结果记录。
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # 逐项处理当前样本、参数或训练批次。
            digest.update(block)  # 完成当前步骤的参数设置、计算或结果记录。
    return digest.hexdigest()  # 返回当前处理结果。


def save_json(path, value):  # 定义本实验的数据处理或运行函数。
    with Path(path).open("x", encoding="utf-8") as stream:  # 完成当前步骤的参数设置、计算或结果记录。
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)  # 完成当前步骤的参数设置、计算或结果记录。


def load_inputs(source, metadata_path, wanted):  # 定义本实验的数据处理或运行函数。
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))  # 完成当前步骤的参数设置、计算或结果记录。
    names = metadata["statistical_columns"]  # 完成当前步骤的参数设置、计算或结果记录。
    physical_names = metadata["physical_columns"]  # 完成当前步骤的参数设置、计算或结果记录。
    if len(names) != 11 or len(set(names)) != 11 or set(physical_names) != set(PHYSICAL_COLUMNS):  # 根据当前数据或运行状态选择处理分支。
        raise ValueError("JSON中的候选特征定义不正确")  # 发现异常时停止，避免输出无效实验结果。
    stat_indices = [names.index(name) for name in SELECTED_STAT_COLUMNS]  # 完成当前步骤的参数设置、计算或结果记录。
    phys_order = [physical_names.index(name) for name in PHYSICAL_COLUMNS]  # 完成当前步骤的参数设置、计算或结果记录。
    result = {}  # 完成当前步骤的参数设置、计算或结果记录。
    with np.load(source, allow_pickle=False) as archive:  # 完成当前步骤的参数设置、计算或结果记录。
        array_ids = archive["array_id"].astype(str)  # 完成当前步骤的参数设置、计算或结果记录。
        total = len(array_ids)  # 完成当前步骤的参数设置、计算或结果记录。
        if array_ids.ndim != 1 or len(set(array_ids)) != total:  # 根据当前数据或运行状态选择处理分支。
            raise ValueError("阵列ID必须唯一")  # 发现异常时停止，避免输出无效实验结果。
        splits = {name: archive[name + "_indices"] for name in ("train", "val", "test")}  # 完成当前步骤的参数设置、计算或结果记录。
        if any(idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer)  # 根据当前数据或运行状态选择处理分支。
               for idx in splits.values()):  # 逐项处理当前样本、参数或训练批次。
            raise ValueError("划分索引必须为非空整数数组")  # 发现异常时停止，避免输出无效实验结果。
        if not np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(total)):  # 根据当前数据或运行状态选择处理分支。
            raise ValueError("训练/验证/测试划分有重叠、遗漏或越界")  # 发现异常时停止，避免输出无效实验结果。

        for key, count in (("stat_mean", 11), ("stat_scale", 11),  # 逐项处理当前样本、参数或训练批次。
                           ("phys_mean", 3), ("phys_scale", 3)):  # 完成当前步骤的参数设置、计算或结果记录。
            value = np.asarray(archive[key], dtype=np.float64)  # 完成当前步骤的参数设置、计算或结果记录。
            if value.shape != (count,) or not np.isfinite(value).all():  # 根据当前数据或运行状态选择处理分支。
                raise ValueError("非法标准化参数：" + key)  # 发现异常时停止，避免输出无效实验结果。
            if "scale" in key and np.any(value <= 0):  # 根据当前数据或运行状态选择处理分支。
                raise ValueError("标准化尺度必须为正")  # 发现异常时停止，避免输出无效实验结果。
            if not np.array_equal(np.asarray(metadata["statistics"][key]), value):  # 根据当前数据或运行状态选择处理分支。
                raise ValueError("NPZ和JSON标准化记录不一致：" + key)  # 发现异常时停止，避免输出无效实验结果。

        mean, scale = archive["phys_mean"], archive["phys_scale"]  # 完成当前步骤的参数设置、计算或结果记录。
        for split_name in wanted:  # 逐项处理当前样本、参数或训练批次。
            idx = splits[split_name]  # 完成当前步骤的参数设置、计算或结果记录。
            stats = archive["statistical"][idx].astype(np.float32)  # 完成当前步骤的参数设置、计算或结果记录。
            phys = archive["physical"][idx].astype(np.float64)  # 完成当前步骤的参数设置、计算或结果记录。
            mask = archive["path_mask"][idx]  # 完成当前步骤的参数设置、计算或结果记录。
            targets = archive["target"][idx].astype(np.float32)  # 完成当前步骤的参数设置、计算或结果记录。
            n = len(idx)  # 完成当前步骤的参数设置、计算或结果记录。
            if (stats.shape != (n, 16, 11) or phys.shape != (n, MAX_PATHS, 3)  # 根据当前数据或运行状态选择处理分支。
                    or mask.shape != (n, MAX_PATHS) or mask.dtype != np.bool_  # 完成当前步骤的参数设置、计算或结果记录。
                    or targets.shape != (n, 3)):  # 完成当前步骤的参数设置、计算或结果记录。
                raise ValueError("NPZ输入形状不符合当前实验定义")  # 发现异常时停止，避免输出无效实验结果。
            if not np.isfinite(stats).all() or not np.isfinite(phys[mask]).all() or not np.isfinite(targets).all():  # 根据当前数据或运行状态选择处理分支。
                raise ValueError("输入含非有限值，请核对数据准备过程")  # 发现异常时停止，避免输出无效实验结果。

            # 统计流：16根天线，每根7项统计量；直接使用NPZ的训练集标准化结果。
            stats = stats[:, :, stat_indices]  # 完成当前步骤的参数设置、计算或结果记录。
            # 保留全部有效路径，按时延升序排列；不筛掉低功率路径。
            ordered = np.zeros_like(phys)  # 完成当前步骤的参数设置、计算或结果记录。
            for row in range(n):  # 逐项处理当前样本、参数或训练批次。
                valid = np.flatnonzero(mask[row])  # 完成当前步骤的参数设置、计算或结果记录。
                delay_column = physical_names.index("delay_ns")  # 完成当前步骤的参数设置、计算或结果记录。
                valid = valid[np.argsort(phys[row, valid, delay_column], kind="stable")]  # 完成当前步骤的参数设置、计算或结果记录。
                ordered[row, :len(valid)] = phys[row, valid]  # 完成当前步骤的参数设置、计算或结果记录。
            valid_counts = mask.sum(axis=1)  # 完成当前步骤的参数设置、计算或结果记录。
            ordered_mask = np.arange(MAX_PATHS)[None, :] < valid_counts[:, None]  # 完成当前步骤的参数设置、计算或结果记录。
            raw = ordered * scale + mean  # 完成当前步骤的参数设置、计算或结果记录。
            raw = raw[:, :, phys_order]  # 完成当前步骤的参数设置、计算或结果记录。
            ordered = ordered[:, :, phys_order]  # 完成当前步骤的参数设置、计算或结果记录。
            # 与当前SDF-Net一致，把两个角度变为sin/cos，保留标准化时延。
            # 仍是3类物理参数，仅展开成5个数值通道，不新增信道特征类别。
            aoa, eoa = np.deg2rad(raw[:, :, 1]), np.deg2rad(raw[:, :, 2])  # 完成当前步骤的参数设置、计算或结果记录。
            path_features = np.stack((ordered[:, :, 0], np.sin(aoa), np.cos(aoa),  # 完成当前步骤的参数设置、计算或结果记录。
                                      np.sin(eoa), np.cos(eoa)), axis=-1)  # 完成当前步骤的参数设置、计算或结果记录。
            path_features[~ordered_mask] = 0  # 完成当前步骤的参数设置、计算或结果记录。

            # 用户确认的单流排列：前16行为统计天线，后15行为全部物理路径。
            # 物理行只有5个数值通道，尾部补2个零；无效路径整行为零。
            # 不把这些列当作复数实部/虚部，也不在这些指纹上执行IFFT。
            packed = np.zeros((n, 1, 16 + MAX_PATHS, 7), dtype=np.float32)  # 保留原CNN的31乘7单流排列，去掉Conv3D专用的末尾单例轴。
            packed[:, 0, :16, :] = stats  # 完成当前步骤的参数设置、计算或结果记录。
            packed[:, 0, 16:, :5] = path_features  # 完成当前步骤的参数设置、计算或结果记录。
            result[split_name] = dict(inputs=packed, target=targets,  # 完成当前步骤的参数设置、计算或结果记录。
                                      array_id=array_ids[idx], valid_path_counts=valid_counts)  # 完成当前步骤的参数设置、计算或结果记录。
    return result  # 返回当前处理结果。


def metrics(prediction, target):  # 定义本实验的数据处理或运行函数。
    errors = np.linalg.norm(prediction.astype(np.float64) - target, axis=1)  # 完成当前步骤的参数设置、计算或结果记录。
    if not np.isfinite(errors).all():  # 根据当前数据或运行状态选择处理分支。
        raise ValueError("预测误差非有限")  # 发现异常时停止，避免输出无效实验结果。
    return dict(n=len(errors), coordinate_mse=float(np.mean((prediction.astype(np.float64) - target) ** 2)),  # 返回当前处理结果。
                mean_error_3d=float(errors.mean()), median_error_3d=float(np.percentile(errors, 50)),  # 完成当前步骤的参数设置、计算或结果记录。
                p90_error_3d=float(np.percentile(errors, 90)))  # 完成当前步骤的参数设置、计算或结果记录。


def evaluate(model, split, device, batch_size):  # 定义本实验的数据处理或运行函数。
    model.eval()  # 完成当前步骤的参数设置、计算或结果记录。
    predictions = []  # 完成当前步骤的参数设置、计算或结果记录。
    with torch.no_grad():  # 完成当前步骤的参数设置、计算或结果记录。
        for start in range(0, len(split["target"]), batch_size):  # 逐项处理当前样本、参数或训练批次。
            inputs = torch.from_numpy(split["inputs"][start:start + batch_size]).to(device)  # 完成当前步骤的参数设置、计算或结果记录。
            predictions.append(model(inputs).cpu().numpy())  # 完成当前步骤的参数设置、计算或结果记录。
    prediction = np.concatenate(predictions)  # 完成当前步骤的参数设置、计算或结果记录。
    return prediction, metrics(prediction, split["target"])  # 返回当前处理结果。


def export_predictions(path, split, prediction):  # 定义本实验的数据处理或运行函数。
    errors = np.linalg.norm(prediction.astype(np.float64) - split["target"], axis=1)  # 完成当前步骤的参数设置、计算或结果记录。
    with path.open("x", encoding="utf-8-sig", newline="") as stream:  # 完成当前步骤的参数设置、计算或结果记录。
        writer = csv.writer(stream)  # 完成当前步骤的参数设置、计算或结果记录。
        writer.writerow(["array_id", "true_x", "true_y", "true_z",  # 完成当前步骤的参数设置、计算或结果记录。
                         "pred_x", "pred_y", "pred_z", "error_3d"])  # 完成当前步骤的参数设置、计算或结果记录。
        for array_id, target, pred, error in zip(split["array_id"], split["target"], prediction, errors):  # 逐项处理当前样本、参数或训练批次。
            writer.writerow([array_id, *target.tolist(), *pred.tolist(), float(error)])  # 完成当前步骤的参数设置、计算或结果记录。


def main():  # 定义本实验的数据处理或运行函数。
    if hasattr(sys.stdout, "reconfigure"):  # 根据当前数据或运行状态选择处理分支。
        sys.stdout.reconfigure(encoding="utf-8")  # 完成当前步骤的参数设置、计算或结果记录。
    parser = argparse.ArgumentParser(description=__doc__)  # 完成当前步骤的参数设置、计算或结果记录。
    parser.add_argument("--data", "--source", dest="data", type=Path,  # 配置命令行选项，支持独立训练、检查或测试。
                        default=Path("/data/prepared.npz"))  # 完成当前步骤的参数设置、计算或结果记录。
    parser.add_argument("--metadata", type=Path)  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--mode", choices=("check", "train", "test"), default="check")  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--output", type=Path)  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--checkpoint", type=Path)  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--epochs", type=int, default=200)  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--seed", type=int, default=42)  # 配置命令行选项，支持独立训练、检查或测试。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")  # 配置命令行选项，支持独立训练、检查或测试。
    args = parser.parse_args()  # 完成当前步骤的参数设置、计算或结果记录。
    if min(args.epochs, args.batch_size, args.patience) < 1 or not np.isfinite(args.lr) or args.lr <= 0:  # 根据当前数据或运行状态选择处理分支。
        parser.error("训练参数必须为正且有限")  # 完成当前步骤的参数设置、计算或结果记录。
    if args.mode == "test" and args.checkpoint is None:  # 根据当前数据或运行状态选择处理分支。
        parser.error("测试必须显式传入 --checkpoint")  # 完成当前步骤的参数设置、计算或结果记录。
    random.seed(args.seed)  # 完成当前步骤的参数设置、计算或结果记录。
    np.random.seed(args.seed)  # 完成当前步骤的参数设置、计算或结果记录。
    torch.manual_seed(args.seed)  # 完成当前步骤的参数设置、计算或结果记录。
    if torch.cuda.is_available():  # 根据当前数据或运行状态选择处理分支。
        torch.cuda.manual_seed_all(args.seed)  # 完成当前步骤的参数设置、计算或结果记录。
    torch.backends.cudnn.benchmark = False  # 完成当前步骤的参数设置、计算或结果记录。
    torch.backends.cudnn.deterministic = True  # 完成当前步骤的参数设置、计算或结果记录。
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device  # 完成当前步骤的参数设置、计算或结果记录。
    if device_name == "cuda" and not torch.cuda.is_available():  # 根据当前数据或运行状态选择处理分支。
        raise ValueError("当前环境没有可用CUDA")  # 发现异常时停止，避免输出无效实验结果。
    device = torch.device(device_name)  # 完成当前步骤的参数设置、计算或结果记录。
    # 在CPU上使用少量线程，避免大量小卷积的线程调度开销。
    if device.type == "cpu":  # 根据当前数据或运行状态选择处理分支。
        torch.set_num_threads(min(4, torch.get_num_threads()))  # 完成当前步骤的参数设置、计算或结果记录。
    metadata_path = args.metadata or args.data.with_suffix(".json")  # 完成当前步骤的参数设置、计算或结果记录。
    wanted = ("test",) if args.mode == "test" else (("train", "val") if args.mode == "train" else ("train",))  # 完成当前步骤的参数设置、计算或结果记录。
    print("读取固定划分：" + ", ".join(wanted), flush=True)  # 显示运行状态或指标，便于确认进度。
    data = load_inputs(args.data, metadata_path, wanted)  # 完成当前步骤的参数设置、计算或结果记录。
    model_config = dict(input_rows=31, input_columns=7, output_dim=3)  # 完成当前步骤的参数设置、计算或结果记录。
    model = AdaptedReConFormer(**model_config).to(device)  # 完成当前步骤的参数设置、计算或结果记录。
    print("输入 [B,1,31,7]；转为[B,7,32]单流CNN序列；全部15个路径槽；输出xyz；设备：" + str(device), flush=True)  # 显示运行状态或指标，便于确认进度。
    print("参数数量：{}".format(sum(value.numel() for value in model.parameters())), flush=True)  # 显示运行状态或指标，便于确认进度。

    if args.mode == "check":  # 根据当前数据或运行状态选择处理分支。
        # 仅用少量训练样本检查前向、反向和梯度，不训练或读取测试特征。
        batch = torch.from_numpy(data["train"]["inputs"][:4]).to(device)  # 完成当前步骤的参数设置、计算或结果记录。
        target = torch.from_numpy(data["train"]["target"][:4]).to(device)  # 完成当前步骤的参数设置、计算或结果记录。
        prediction = model(batch)  # 完成当前步骤的参数设置、计算或结果记录。
        loss = (prediction - target).square().mean()  # 完成当前步骤的参数设置、计算或结果记录。
        loss.backward()  # 完成当前步骤的参数设置、计算或结果记录。
        if prediction.shape != (len(batch), 3) or not torch.isfinite(prediction).all() or not torch.isfinite(loss):  # 根据当前数据或运行状态选择处理分支。
            raise ValueError("前向结果异常")  # 发现异常时停止，避免输出无效实验结果。
        expected_unused = {"backbone.rcnn.conv01.weight", "backbone.rcnn.conv02.weight", "backbone.rcnn.bn0.0.weight", "backbone.rcnn.bn0.0.bias", "backbone.rcnn.bn0.1.weight", "backbone.rcnn.bn0.1.bias"}  # 作者默认CNN循环一次时这些辅助分支没有被调用。
        unexpected_unused = [name for name, value in model.named_parameters() if value.grad is None and name not in expected_unused]  # 只允许源码可解释的未使用参数。
        if unexpected_unused or any(value.grad is not None and not torch.isfinite(value.grad).all() for value in model.parameters()):  # 根据当前数据或运行状态选择处理分支。
            raise ValueError("梯度缺失或非有限")  # 发现异常时停止，避免输出无效实验结果。
        print("检查通过：真实训练样本前向/反向和三维输出有效；仅作者未调用的辅助分支没有梯度。", flush=True)  # 显示运行状态或指标，便于确认进度。
        return  # 完成当前步骤的参数设置、计算或结果记录。

    output = args.output or ROOT / "runs" / ("seed{}_".format(args.seed) + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))  # 完成当前步骤的参数设置、计算或结果记录。
    output.mkdir(parents=True, exist_ok=False)  # 完成当前步骤的参数设置、计算或结果记录。
    source_hashes = {}  # 完成当前步骤的参数设置、计算或结果记录。
    for name in ("model.py", "run.py", "models/__init__.py", "models/RecurrentRes.py", "models/RConFormer_runtime.py", "upstream_source.json"):  # 逐项处理当前样本、参数或训练批次。
        source = ROOT / name  # 完成当前步骤的参数设置、计算或结果记录。
        if source.exists():  # 根据当前数据或运行状态选择处理分支。
            source_hashes[name] = sha256(source)  # 完成当前步骤的参数设置、计算或结果记录。
            (output / name).parent.mkdir(parents=True, exist_ok=True)  # 完成当前步骤的参数设置、计算或结果记录。
            shutil.copy2(source, output / name)  # 完成当前步骤的参数设置、计算或结果记录。
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},  # 完成当前步骤的参数设置、计算或结果记录。
                  source_sha256=sha256(args.data), metadata_sha256=sha256(metadata_path),  # 完成当前步骤的参数设置、计算或结果记录。
                  packing_version=PACKING_VERSION, statistical_features=SELECTED_STAT_COLUMNS,  # 完成当前步骤的参数设置、计算或结果记录。
                  physical_parameters=PHYSICAL_COLUMNS,  # 完成当前步骤的参数设置、计算或结果记录。
                  physical_input_columns=["delay_z", "sin_AoA", "cos_AoA", "sin_EoA", "cos_EoA"],  # 完成当前步骤的参数设置、计算或结果记录。
                  model_config=model_config, source_hashes=source_hashes, upstream=UPSTREAM,  # 完成当前步骤的参数设置、计算或结果记录。
                  validation_policy="early stopping: validation MSE; checkpoint: validation mean 3D error",  # 完成当前步骤的参数设置、计算或结果记录。
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__))  # 完成当前步骤的参数设置、计算或结果记录。
    save_json(output / "config.json", config)  # 完成当前步骤的参数设置、计算或结果记录。

    if args.mode == "test":  # 根据当前数据或运行状态选择处理分支。
        # 只加载自己产生且来源可信的检查点；数据和排列方式必须与训练一致。
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)  # 完成当前步骤的参数设置、计算或结果记录。
        if saved["source_sha256"] != config["source_sha256"] or saved["metadata_sha256"] != config["metadata_sha256"]:  # 根据当前数据或运行状态选择处理分支。
            raise ValueError("测试数据或元数据与训练记录不一致")  # 发现异常时停止，避免输出无效实验结果。
        if saved["packing_version"] != PACKING_VERSION or saved["model_config"] != model_config:  # 根据当前数据或运行状态选择处理分支。
            raise ValueError("检查点输入排列或网络配置不一致")  # 发现异常时停止，避免输出无效实验结果。
        model.load_state_dict(saved["model"])  # 完成当前步骤的参数设置、计算或结果记录。
        prediction, report = evaluate(model, data["test"], device, args.batch_size)  # 完成当前步骤的参数设置、计算或结果记录。
        export_predictions(output / "test_predictions.csv", data["test"], prediction)  # 完成当前步骤的参数设置、计算或结果记录。
        save_json(output / "test_metrics.json", report)  # 完成当前步骤的参数设置、计算或结果记录。
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)  # 显示运行状态或指标，便于确认进度。
        print("结果目录：" + str(output), flush=True)  # 显示运行状态或指标，便于确认进度。
        return  # 完成当前步骤的参数设置、计算或结果记录。

    # 本适配版沿用其他比较实验的Adam设置，不声称使用作者原活动分类训练协议。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)  # 完成当前步骤的参数设置、计算或结果记录。
    generator = torch.Generator().manual_seed(args.seed)  # 完成当前步骤的参数设置、计算或结果记录。
    loader = DataLoader(TensorDataset(torch.from_numpy(data["train"]["inputs"]),  # 完成当前步骤的参数设置、计算或结果记录。
                                     torch.from_numpy(data["train"]["target"])),  # 完成当前步骤的参数设置、计算或结果记录。
                        batch_size=args.batch_size, shuffle=True, generator=generator,  # 完成当前步骤的参数设置、计算或结果记录。
                        drop_last=False, num_workers=0)  # 完成当前步骤的参数设置、计算或结果记录。
    best_error, best_early_loss, stale = float("inf"), float("inf"), 0  # 完成当前步骤的参数设置、计算或结果记录。
    with (output / "history.csv").open("x", encoding="utf-8", newline="") as stream:  # 完成当前步骤的参数设置、计算或结果记录。
        writer = csv.writer(stream)  # 完成当前步骤的参数设置、计算或结果记录。
        writer.writerow(["epoch", "train_loss", "val_mse", "val_loss", "val_mean_error_3d"])  # 完成当前步骤的参数设置、计算或结果记录。
        for epoch in range(1, args.epochs + 1):  # 逐项处理当前样本、参数或训练批次。
            model.train()  # 完成当前步骤的参数设置、计算或结果记录。
            accumulated, count = 0.0, 0  # 完成当前步骤的参数设置、计算或结果记录。
            for inputs, target in loader:  # 逐项处理当前样本、参数或训练批次。
                inputs, target = inputs.to(device), target.to(device)  # 完成当前步骤的参数设置、计算或结果记录。
                optimizer.zero_grad(set_to_none=True)  # 完成当前步骤的参数设置、计算或结果记录。
                loss = (model(inputs) - target).square().mean()  # 监督训练循环CNN、Transformer和三维回归头。
                if not torch.isfinite(loss):  # 根据当前数据或运行状态选择处理分支。
                    raise ValueError("训练损失非有限")  # 发现异常时停止，避免输出无效实验结果。
                loss.backward()  # 完成当前步骤的参数设置、计算或结果记录。
                optimizer.step()  # 完成当前步骤的参数设置、计算或结果记录。
                accumulated += float(loss.detach()) * len(target)  # 完成当前步骤的参数设置、计算或结果记录。
                count += len(target)  # 完成当前步骤的参数设置、计算或结果记录。
            _, report = evaluate(model, data["val"], device, args.batch_size)  # 完成当前步骤的参数设置、计算或结果记录。
            val_loss = report["coordinate_mse"]  # 本协议不增加额外卷积核L2项，验证损失即坐标MSE。
            writer.writerow([epoch, accumulated / count, report["coordinate_mse"], val_loss, report["mean_error_3d"]])  # 完成当前步骤的参数设置、计算或结果记录。
            stream.flush()  # 完成当前步骤的参数设置、计算或结果记录。
            print("epoch {:3d} train={:.6f} val_loss={:.6f} val_mean={:.6f} m".format(  # 显示运行状态或指标，便于确认进度。
                epoch, accumulated / count, val_loss, report["mean_error_3d"]), flush=True)  # 完成当前步骤的参数设置、计算或结果记录。
            if report["mean_error_3d"] < best_error:  # 根据当前数据或运行状态选择处理分支。
                best_error = report["mean_error_3d"]  # 完成当前步骤的参数设置、计算或结果记录。
                torch.save(dict(model=model.state_dict(), model_config=model_config,  # 完成当前步骤的参数设置、计算或结果记录。
                                source_sha256=config["source_sha256"], metadata_sha256=config["metadata_sha256"],  # 完成当前步骤的参数设置、计算或结果记录。
                                packing_version=PACKING_VERSION, epoch=epoch, seed=args.seed,  # 完成当前步骤的参数设置、计算或结果记录。
                                validation=report), output / "best.pt")  # 完成当前步骤的参数设置、计算或结果记录。
            if val_loss < best_early_loss:  # 根据当前数据或运行状态选择处理分支。
                best_early_loss, stale = val_loss, 0  # 完成当前步骤的参数设置、计算或结果记录。
            else:  # 完成当前步骤的参数设置、计算或结果记录。
                stale += 1  # 完成当前步骤的参数设置、计算或结果记录。
            if stale >= args.patience:  # 根据当前数据或运行状态选择处理分支。
                print("验证损失连续{}轮未改善，停止。".format(args.patience), flush=True)  # 显示运行状态或指标，便于确认进度。
                break  # 完成当前步骤的参数设置、计算或结果记录。
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)  # 完成当前步骤的参数设置、计算或结果记录。
    model.load_state_dict(saved["model"])  # 完成当前步骤的参数设置、计算或结果记录。
    prediction, report = evaluate(model, data["val"], device, args.batch_size)  # 完成当前步骤的参数设置、计算或结果记录。
    export_predictions(output / "validation_predictions.csv", data["val"], prediction)  # 完成当前步骤的参数设置、计算或结果记录。
    save_json(output / "validation_metrics.json", dict(best_epoch=saved["epoch"], **report))  # 完成当前步骤的参数设置、计算或结果记录。
    print("训练完成；测试需显式运行 --mode test。结果目录：" + str(output), flush=True)  # 显示运行状态或指标，便于确认进度。


if __name__ == "__main__":  # 根据当前数据或运行状态选择处理分支。
    main()  # 完成当前步骤的参数设置、计算或结果记录。
