# 中文阅读路线：本文件负责数据读取、输入整理、训练、验证和显式测试。
# 一次训练的数据流（N=所选集合样本数，B=批大小，K=路径上限，D=模型宽度）：
# prepared.npz 中的固定 train_indices / val_indices
#   ├─ statistical [N,16,11] → 固定取8列 → [N,8,4,4]
#   ├─ physical [N,15,3] → 有效路径筛选/排序 → 圆周编码 [N,K,5]
#   ├─ path_mask [N,15] → 对应的有效路径标记 [N,K]
#   └─ target [N,3] → 监督标签，不作为定位网络输入
# TensorDataset → DataLoader → SDFNet.forward(s,p,mask,return_aux=True)
#   → 主预测 MSE + 可选距离损失 + 可选双流辅助损失
#   → backward → 梯度裁剪 → optimizer.step → 可选 EMA
# 每轮只用验证集选择 best.pt；耐心耗尽或达到轮数上限后停止。
# 训练结束恢复 best.pt，导出训练/验证指标；测试需单独 --mode test。
# 本脚本不重新运行特征筛选，也不从配套 JSON 动态推断统计特征列序。
# 参数实际值以命令行和运行目录 config.json 为准；本次仅添加说明，不改默认值。

"""Independent SDF-Net experiment. Default training selects on validation only.

Run final test explicitly with --mode test --checkpoint .../best.pt.
No raw CSI/CIR, coordinate inputs, test-guided fitting or split changes.
"""
import argparse  # 解析命令行参数及生成帮助信息
import copy  # 深拷贝模型以建立可选EMA副本
import csv  # 读写训练日志、预测结果及CDF表格
import hashlib  # 计算数据、配置与源码的SHA256哈希
import json  # 读取/保存配置和指标JSON
import math  # 计算有限性检查、余弦学习率等数学表达式
import random  # 设置Python随机种子
import shutil  # 复制源码、启动器及预处理配置快照
import sys  # 访问当前Python解释器、模块搜索路径和命令行环境
import time  # 记录训练耗时
from datetime import datetime  # 生成运行目录使用的时间戳
from pathlib import Path  # 使用Path管理数据、源码和输出路径

import numpy as np  # 读取NPZ并执行输入整理、统计与指标计算

EXPERIMENT = Path(__file__).resolve().parent  # 当前脚本所在目录；直接运行与使用历史运行快照时，此路径会不同
PROJECT = next((p for p in EXPERIMENT.parents if (p / "common" / "data.py").is_file()),  # 向父目录寻找旧版项目标记 common/data.py；这里仅确定路径，不导入该文件
               EXPERIMENT.parents[1])  # 找不到旧版标记时使用备用项目目录；显式 --data 可指定实际数据位置
ARCHITECTURE = "sdfnet_full_selected8_external_input_weighting_v3"  # 架构版本标识，测试时用于拒绝不兼容的检查点
SOURCE_STAT_COLUMNS = ["Rx_power_dBm", "R_max_dB", "Rise_time_ns", "Tau_MED_ns",  # 声明 NPZ 统计数组的固定11列顺序，名称与数值含义需要在数据准备阶段对应
                       "Tau_RMS_ns", "Kurtosis", "Skewness", "K_factor", "PL_dB",  # 原始列序的中间部分；这里只声明名称，不交换已有数组列
                       "DPR", "tau_dom_RMS_ns"]  # 原始统计列序的最后两列
STAT_COLUMNS = ["tau_dom_RMS_ns", "K_factor", "PL_dB",  # 声明完整模型实际读取的8个统计特征及其网络通道顺序
                "Rx_power_dBm", "Tau_MED_ns", "DPR", "R_max_dB"]  # 8个输入统计通道的后4个；其余3个统计候选不进入本模型
STAT_INDICES = [SOURCE_STAT_COLUMNS.index(c) for c in STAT_COLUMNS]  # 按名称查得固定列索引，实际取数仍按这些索引执行
SOURCE_FILES = ("model.py", "run.py")  # 需要计算哈希、随运行记录核对的两份核心源码


def sha256(path):  # 计算文件 SHA256，用于核对数据与代码版本
    digest = hashlib.sha256()  # 创建 SHA256 摘要对象
    with Path(path).open("rb") as handle:  # 以二进制读取文件，避免编码或换行转换影响哈希
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):  # 每次读取1 MiB，直到读到空字节串
            digest.update(chunk)  # 把当前文件块计入摘要
    return digest.hexdigest()  # 返回十六进制哈希字符串


def save_json(path, value):  # 把配置或指标字典保存为 UTF-8 JSON
    with Path(path).open("w", encoding="utf-8") as handle:  # 打开目标 JSON 文件用于写入
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)  # 缩进保存并保留中文；拒绝把 NaN/无穷值写成指标

# 1. 这里是训练的入口，读取训练集和验证集
def read_prepared(path, active_splits=("train", "val")):  # 读取指定的数据集合，默认仅返回训练集与验证集
    """Use the existing array-level split; validate only requested feature rows."""  # 沿用NPZ已有阵列级划分，只对本次请求集合的特征行做数值验证
    if not active_splits or not set(active_splits).issubset({"train", "val", "test"}):  # 要求至少选择一个集合，且集合名只能是 train/val/test
        raise ValueError("Invalid active splits")  # 集合参数错误时停止读取
    with np.load(path, allow_pickle=False) as archive:  # 打开 NPZ，禁止加载依赖 pickle 的对象数组
        n = len(archive["array_id"])  # 由阵列编号数获得全部阵列样本数
        index = {name: archive[name + "_indices"] for name in ("train", "val", "test")}  # 读取已固定的三组索引，本脚本不重新随机划分
        for name, idx in index.items():  # 逐个检查训练/验证/测试索引
            if idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer):  # 集合索引必须是一维、非空的整数数组
                raise ValueError("Invalid split indices: " + name)  # 报告不合法的索引集合
        if not np.array_equal(np.sort(np.concatenate(list(index.values()))), np.arange(n)):  # 三组索引合起来必须恰好覆盖所有样本一次，防止重叠、遗漏或越界
            raise ValueError("Prepared splits overlap, omit samples or contain invalid indices")  # 划分覆盖关系异常时终止
        ids = archive["array_id"]  # 读取每个阵列样本的唯一标识
        if len(set(ids.tolist())) != n:  # 检查阵列编号是否重复
            raise ValueError("Duplicate array IDs")  # 重复编号会破坏结果对齐，因此报错
        keys = ("statistical", "physical", "path_mask", "path_power_db", "target", "array_id")  # 本次输入整理和监督学习需要的 NPZ 数据字段
        shapes = dict(statistical=(n, 16, 11), physical=(n, 15, 3),  #### 2 统计为16天线×11特征；物理为最多15路径×3参数
                      path_mask=(n, 15), path_power_db=(n, 15), target=(n, 3), array_id=(n,))  # 同时约定路径掩码、路径功率、三维标签和样本编号形状
        data = {name: {} for name in active_splits}  # 为实际请求的集合创建结果字典
        for key in keys:  # 逐项读取所需数组
            values = archive[key]  # 取出当前字段的数据
            if values.shape != shapes[key]:  # 检查数组维度是否满足约定
                raise ValueError("Unexpected prepared shape for " + key)  # 字段形状异常时指出字段名并停止
            for name in active_splits:  # 仅为调用者要求的集合提取数据
                data[name][key] = values[index[name]]  # 使用 NPZ 已保存的索引按样本轴取行
        mean, scale = archive["phys_mean"], archive["phys_scale"]  # 读取物理参数标准化时使用的均值和尺度，供角度恢复与编码使用
    if mean.shape != (3,) or scale.shape != (3,) or not np.isfinite(mean).all():  # 物理均值/尺度都应有3项，且均值必须有限
        raise ValueError("Invalid prepared physical normalization")  # 标准化参数形状或均值异常则报错
    if not np.isfinite(scale).all() or np.any(scale <= 0):  # 物理尺度必须有限且严格大于零
        raise ValueError("Invalid physical standard deviations")  # 拒绝零尺度、负尺度或非有限尺度
    for name, split in data.items():  # 逐个验证实际读取的集合
        if split["path_mask"].dtype != np.bool_:  # 路径掩码必须是布尔值，True 表示有效路径
            raise ValueError("Expected boolean path masks")  # 掩码类型错误时终止
        for key in ("statistical", "target"):  # 统计输入和坐标标签都要求是有限数值
            if not np.isfinite(split[key]).all():  # 检查这些字段中是否存在 NaN 或无穷值
                raise ValueError("Nonfinite prepared {} in {}".format(key, name))  # 说明哪个集合的哪个字段包含异常数值
        if not np.isfinite(split["physical"][split["path_mask"]]).all():  # 只检查有效物理路径；无效填充路径由掩码处理
            raise ValueError("Nonfinite real path features: rerun agreed mean imputation")  # 有效路径包含异常值时拒绝继续训练
    return data, dict(phys_mean=mean.tolist(), phys_scale=scale.tolist())  # 返回各集合数据及物理标准化参数，统计输入沿用 NPZ 已处理的数值

# 3 将它在16根天线上的数值恢复成4×4网格 [N,16,8] → [N,4,4,8] → [N,8,4,4]
def pack_input(split, input_config):  # 把一组阵列样本整理成统计网格、物理路径和掩码三路模型参数
    """Split selected fingerprints into a 4x4 statistical grid and K path tokens."""  # 将选定统计特征排列成4×4网格，并把有效物理路径整理成K个token
    k = input_config["max_paths"]  # 读取本次最多保留多少条路径
    encoding = input_config["angle_encoding"]  # 读取角度编码方式：标量或正余弦圆周编码
    if not 1 <= k <= 15 or encoding not in ("scalar", "circular"):  # 检查路径上限与编码模式是否合法
        raise ValueError("Invalid input configuration")  # 输入配置异常时停止
    n = len(split["target"])  # 仅用标签长度获取样本数，真实坐标不拼入网络输入
    stats = split["statistical"][:, :, STAT_INDICES]  # 3 从 [N,16,11] 固定取出8个统计通道，得到 [N,16,8] ### 11->8
    grid = np.ascontiguousarray(stats.reshape(n, 4, 4, len(STAT_COLUMNS)).transpose(0, 3, 1, 2), dtype=np.float32)  # 把16天线恢复成4×4阵列，再把8个特征移到通道轴，得到 [N,8,4,4]
    physical = np.zeros((n, k, 3), dtype=np.float32)  # 建立最多K条路径的3参数容器，不足的路径先置零
    mask = np.zeros((n, k), dtype=bool)  # 建立全无效掩码，后续只为真实路径设 True
    for i in range(n):  # 按阵列逐个整理路径，因为各样本的有效路径数可能不同
        indices = np.flatnonzero(split["path_mask"][i])  # 找到当前阵列的所有有效路径索引
        if len(indices) > k:  # 仅当有效路径多于K条时，才需要按功率截取
            powers = split["path_power_db"][i, indices]  # 读取有效路径功率；功率用于选路径，不直接作为网络输入通道
            if not np.isfinite(powers).all():  # 按功率选择前，要求真实路径功率均有限
                raise ValueError("Cannot choose strongest K: invalid real-path power for " + str(split["array_id"][i]))  # 功率异常时报告阵列编号，避免静默选错路径
            indices = indices[np.argsort(-powers, kind="stable")[:k]]  # 按功率降序稳定排序，保留最强K条有效路径
        indices = indices[np.argsort(split["physical"][i, indices, 0], kind="stable")]  # 把保留路径按第0列时延升序排列，形成一维卷积的路径序列
        physical[i, :len(indices)] = split["physical"][i, indices]  # 将有效路径依次写入容器，保留已有标准化参数值
        mask[i, :len(indices)] = True  # 将已写入的路径标为有效，其余仍是填充位置
    if encoding == "scalar":  # 标量模式保留3个标准化物理参数
        features = physical[:, :, [1, 0, 2]]  # 将原来的 delay/AoA/EoA 调整为 AoA/delay/EoA，匹配标量基线约定
    else:  # 4 圆周编码先恢复原角度，再将两个角度编码为正弦与余弦 物理输入 p：   [B,15,5]
        mean = np.asarray(input_config["phys_mean"], dtype=np.float64)  # 取出三个物理参数的原始标准化均值
        scale = np.asarray(input_config["phys_scale"], dtype=np.float64)  # 取出对应标准化尺度
        raw = physical.astype(np.float64) * scale + mean  # 恢复原单位物理参数，以便对角度进行正确的三角函数编码
        aoa, eoa = np.deg2rad(raw[:, :, 1]), np.deg2rad(raw[:, :, 2])  # 把以度为单位的 AoA/EoA 转成弧度
        features = np.stack((physical[:, :, 0], np.sin(aoa), np.cos(aoa),  # 前3个编码通道依次为标准化时延、AoA正弦、AoA余弦
                             np.sin(eoa), np.cos(eoa)), axis=-1)  # 再加入 EoA 正弦/余弦，形成 [N,K,5]；这是编码展开而非新增测量参数
    features[~mask] = 0  # 编码完成后再次把填充路径所有通道置零
    features = np.ascontiguousarray(features, dtype=np.float32)  # 转为连续 float32 内存，便于直接创建 PyTorch 张量
    if not np.isfinite(grid).all() or not np.isfinite(features).all():  # 确认最终两路输入均没有 NaN/无穷值
        raise ValueError("Nonfinite packed inputs")  # 最终输入异常时阻止进入网络
    return dict(statistical=grid, physical=features, path_mask=mask)  # 返回统计 [N,8,4,4]、物理 [N,K,3或5]、有效标记 [N,K]


def fit_target_transform(target):  # 仅根据训练坐标拟合中心与全局尺度，供模型稳定回归与反变换
    """One scalar scale keeps xyz MSE weighting unchanged (no per-axis rescale)."""  # 三个坐标轴共用一个尺度，避免逐轴缩放改变坐标MSE的相对权重
    target = np.asarray(target, dtype=np.float64)  # 用双精度计算坐标变换参数
    center = target.mean(axis=0)  # 分别计算训练坐标 x/y/z 的均值，得到3维中心
    scale = max(float(np.sqrt(np.mean((target - center) ** 2))), 1e-6)  # 计算三个轴共享的标量尺度并设下限，保持各轴损失相对权重不变
    return center.astype(np.float32), scale  # 返回坐标中心和单个正尺度，后续保存在模型 buffer 中


def metrics(pred, target):  # 在原始坐标单位下计算3D误差、坐标MSE及分位数等指标
    delta = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)  # 逐样本计算预测坐标减真实坐标，得到 [N,3] 的偏差
    if delta.ndim != 2 or delta.shape[1] != 3 or not len(delta) or not np.isfinite(delta).all():  # 要求偏差非空、每行3维且数值有限
        raise ValueError("Expected finite nonempty xyz predictions and labels")  # 输入预测或标签不满足指标约定时停止
    error = np.linalg.norm(delta, axis=1)  # 每行求欧氏距离，即每个样本的3D定位误差
    return dict(n=len(error), coordinate_mse=float(np.mean(delta ** 2)),  # 保存样本数及所有样本/坐标轴上的平均平方误差
                mean_error_3d=float(error.mean()), median_error_3d=float(np.median(error)),  # 保存3D欧氏误差均值与中位数；均值不是坐标MSE
                rmse_3d=float(np.sqrt(np.mean(error ** 2))), std_error_3d=float(error.std()),  # 保存3D距离RMSE及样本间误差的总体标准差，不是跨种子的标准差
                p90_error_3d=float(np.percentile(error, 90)), p95_error_3d=float(np.percentile(error, 95)),  # 保存90%和95%误差分位数
                axis_mae=dict(zip(("x", "y", "z"), np.abs(delta).mean(0).tolist())),  # 分别计算x/y/z三个坐标轴的平均绝对偏差
                fraction_within_0_5=float(np.mean(error <= 0.5)),  # 计算误差不超过0.5个坐标单位的样本比例
                fraction_within_1=float(np.mean(error <= 1)),  # 计算误差不超过1个坐标单位的样本比例
                fraction_within_2=float(np.mean(error <= 2)))  # 计算误差不超过2个坐标单位的样本比例并返回指标字典


def export_predictions(output, name, ids, pred, target, gates):  # 导出逐样本预测、真实坐标、门控权重和经验CDF数据
    errors = np.linalg.norm(np.asarray(pred, dtype=np.float64) - target, axis=1)  # 计算每个阵列位置的3D欧氏误差
    with (output / (name + "_predictions.csv")).open("w", encoding="utf-8-sig", newline="") as handle:  # 创建带 UTF-8 BOM 的预测CSV，便于表格软件读取
        writer = csv.writer(handle)  # 创建CSV写入器
        writer.writerow(["array_id", "true_x", "true_y", "true_z", "pred_x", "pred_y", "pred_z",  # 写入样本编号、真实坐标与预测坐标列名
                         "error_3d", "stat_gate", "physical_gate"])  # 继续写入3D误差与两路门控权重列名
        for sample_id, truth, estimate, error, gate in zip(ids, target, pred, errors, gates):  # 保持样本顺序一致，逐行导出结果
            writer.writerow([sample_id] + truth.tolist() + estimate.tolist() + [float(error)] + gate.tolist())  # 合并当前样本的编号、坐标、误差和权重后写出
    with (output / (name + "_cdf.csv")).open("w", encoding="utf-8", newline="") as handle:  # 单独创建经验CDF数据文件
        writer = csv.writer(handle)  # 创建CSV写入器
        writer.writerow(["error_3d", "cumulative_probability"])  # 写入误差与累计概率两个列名
        writer.writerows(zip(np.sort(errors), np.arange(1, len(errors) + 1) / len(errors)))  # 将排序后的误差与 i/N 配对，得到经验CDF


def load_stat_weight_file(path):  # 读取固定统计输入倍率JSON，并严格核对特征名称与顺序
    """Read named input multipliers; never silently guess feature order."""  # 读取按名称保存的倍率，特征顺序不匹配时明确报错
    payload = Path(path).read_bytes()  # 保留配置文件的原始字节，既用于解析也用于保存快照与哈希
    obj = json.loads(payload.decode("utf-8-sig"))  # 兼容带BOM和普通UTF-8的JSON配置
    if obj.get("statistical_features") != STAT_COLUMNS:  # 要求配置中的特征列表与本模型8通道顺序完全一致
        raise ValueError("Weight file feature names/order differ from STAT_COLUMNS")  # 名称或顺序不一致时拒绝猜测匹配关系
    mapping = obj.get("feature_weights")  # 读取特征名称到倍率的映射
    if not isinstance(mapping, dict) or set(mapping) != set(STAT_COLUMNS):  # 映射必须恰好包含当前8个统计特征
        raise ValueError("Weight file must contain exactly the selected feature names")  # 倍率字典缺列或多列时停止
    weights = [mapping[name] for name in STAT_COLUMNS]  # 按网络实际通道顺序排列倍率
    if any(isinstance(v, bool) or not isinstance(v, (int, float))  # 排除布尔值和非数值类型，布尔值不应被当作倍率1或0
           or not math.isfinite(v) or v <= 0 for v in weights):  # 进一步要求每个倍率有限且大于零
        raise ValueError("Input weights must be finite positive numbers")  # 非法倍率会导致输入错误，因此报错
    provenance = dict(file=str(Path(path).resolve()), sha256=hashlib.sha256(payload).hexdigest(),  # 记录倍率配置的绝对路径及原始内容哈希
                      training_run=obj.get("source_training_run"),  # 保留倍率来源的历史训练运行标识
                      selection=obj.get("source_selection", "Unspecified; inspect supplied configuration"))  # 保留来源模型的选择方式；缺失时明确标为未说明
    return weights, provenance, payload  # 返回有序倍率、来源信息和原始JSON字节


def export_stat_feature_weights(output, model):  # 导出当前模型实际使用的输入倍率，通常在恢复最佳检查点之后调用
    layer = model.stat_feature_weighting  # 取得网络最前面的逐统计通道加权层
    weights = layer.get_weights().detach().cpu().tolist()  # 从计算图分离倍率，移到CPU并转换为可序列化列表
    save_json(output / "stat_feature_weights.json", dict(  # 创建实际输入倍率记录，不改变模型或输入数值
        mode=layer.mode,  # 记录 none/learned/fixed 三种模式中的实际模式
        parameterization=("fixed multipliers from prior trained model" if layer.mode=="fixed"  # 固定模式的倍率来自外部配置所指向的既有训练结果
                          else "1 + 0.5 * tanh(logits); initialized to 1" if layer.mode=="learned"  # 学习模式从倍率1开始，以双曲正切约束变化范围
                          else "identity multipliers"),  # none模式的倍率均为1，输入原样通过
        fixed_weight_source=layer.fixed_weight_source,  # 记录固定倍率来源；非固定模式通常为None
        bounds=([0.5, 1.5] if layer.mode=="learned" else None),  # 学习模式理论倍率落在0.5与1.5之间
        statistical_features=STAT_COLUMNS,  # 同时保存特征顺序，防止只看数值时错配
        feature_weights=dict(zip(STAT_COLUMNS, weights)),  # 把实际倍率按特征名称保存
        checkpoint="best validation checkpoint",  # 说明此处导出的倍率对应验证集选择的最佳检查点
        interpretation="Input multipliers; not standalone causal feature importance"))  # 输入倍率不等同于独立的因果重要性结论


def build_parser():  # 声明run.py接受的所有命令行选项及默认值
    p = argparse.ArgumentParser(description=__doc__)  # 创建命令行解析器，帮助文本使用模块说明
    p.add_argument("--data", type=Path, default=PROJECT / "data" / "prepared.npz")  # 命令行参数 --data：输入prepared.npz路径，沿用其中固定的数据划分；此入口默认 PROJECT / "data" / "prepared.npz"
    p.add_argument("--output", type=Path, help="Must be a new directory")  # 命令行参数 --output：本次训练/测试的独立新输出目录
    p.add_argument("--mode", choices=("train", "test"), default="train")  # 命令行参数 --mode：train执行训练与验证；test读取检查点进行显式测试；此入口默认 "train"
    p.add_argument("--checkpoint", type=Path)  # 命令行参数 --checkpoint：测试使用的best.pt路径，本实现不支持用它续训
    p.add_argument("--skip-test", action="store_true", help="Compatibility flag: training always skips test")  # 命令行参数 --skip-test：兼容旧命令的标志；当前训练流程始终跳过测试
    p.add_argument("--epochs", type=int, default=200)  # 命令行参数 --epochs：最大训练轮数，早停可能提前结束；此入口默认 200
    p.add_argument("--batch-size", type=int, default=64)  # 命令行参数 --batch-size：每批包含的阵列位置数，不是天线数或路径数；此入口默认 64
    p.add_argument("--lr", type=float, default=1e-3)  # 命令行参数 --lr：优化器初始学习率，会受调度器影响；此入口默认 1e-3
    p.add_argument("--weight-decay", type=float, default=1e-4)  # 命令行参数 --weight-decay：优化器的权重衰减系数；此入口默认 1e-4
    p.add_argument("--patience", type=int, default=40)  # 命令行参数 --patience：连续多少轮验证指标未改善后早停；0关闭早停；此入口默认 40
    p.add_argument("--seed", type=int, default=42)  # 命令行参数 --seed：本次初始化、随机增强和训练打乱顺序使用的随机种子；此入口默认 42
    p.add_argument("--device", default="auto")  # 命令行参数 --device：训练/测试设备，auto优先CUDA，也可显式指定cuda或cpu；此入口默认 "auto"
    p.add_argument("--num-workers", type=int, default=0)  # 命令行参数 --num-workers：DataLoader加载数据的工作进程数，0表示主进程加载；此入口默认 0
    p.add_argument("--max-paths", type=int, default=15)  # 命令行参数 --max-paths：每个样本最多保留的有效物理路径数，范围1到15；此入口默认 15
    p.add_argument("--angle-encoding", choices=("circular", "scalar"), default="circular")  # 命令行参数 --angle-encoding：circular把两个角度展开为正余弦；scalar保留3维标量；此入口默认 "circular"
    p.add_argument("--cnn-width", type=int, default=32)  # 命令行参数 --cnn-width：两路CNN内部的特征通道宽度；此入口默认 32
    p.add_argument("--cnn-blocks", type=int, default=2, help="0 removes CNN residual blocks")  # 命令行参数 --cnn-blocks：两路各自的残差CNN块数；0只保留输入映射；此入口默认 2
    p.add_argument("--d-model", type=int, default=96)  # 命令行参数 --d-model：共享Transformer中每个token的维度，必须能被头数整除；此入口默认 96
    p.add_argument("--num-heads", type=int, default=4)  # 命令行参数 --num-heads：Transformer每层MHSA的注意力头数；此入口默认 4
    p.add_argument("--num-layers", type=int, default=2, help="0 removes Transformer blocks")  # 命令行参数 --num-layers：共享Transformer编码层数，0跳过编码层；此入口默认 2
    p.add_argument("--ffn-ratio", type=int, default=2)  # 命令行参数 --ffn-ratio：Transformer前馈隐藏宽度相对d_model的倍率；此入口默认 2
    p.add_argument("--dropout", type=float, default=0.1)  # 命令行参数 --dropout：训练时随机丢弃概率；评估模式关闭丢弃；此入口默认 0.1
    p.add_argument("--fusion", choices=("gated", "concat"), default="gated")  # 命令行参数 --fusion：gated按样本学习两路倍率；concat不使用学习门控；此入口默认 "gated"
    p.add_argument("--pooling", choices=("attention", "mean"), default="attention")  # 命令行参数 --pooling：attention用可学习分数池化，mean对有效token取平均；此入口默认 "attention"
    p.add_argument("--no-coordinate-attention", action="store_true",  # 命令行参数 --no-coordinate-attention：传入此开关才关闭统计流CNN后的行列CA
                   help="Disable Coordinate Attention on the statistical CNN feature map")  # 帮助文本说明该开关仅关闭统计流CNN后的CA
    p.add_argument("--branch", choices=("both", "stat", "physical"), default="both")  # 命令行参数 --branch：both启用双流；stat或physical关闭另一分支进行消融；此入口默认 "both"
    p.add_argument("--no-shortcut", action="store_true")  # 命令行参数 --no-shortcut：传入此开关才关闭早期双流摘要到坐标的local_head捷径
    p.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")  # 命令行参数 --optimizer：选择AdamW或Adam进行参数更新；此入口默认 "adamw"
    p.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")  # 命令行参数 --scheduler：cosine为预热后余弦衰减，none为固定学习率；此入口默认 "cosine"
    p.add_argument("--warmup-epochs", type=int, default=10)  # 命令行参数 --warmup-epochs：学习率线性预热轮数；此入口默认 10
    p.add_argument("--min-lr-ratio", type=float, default=0.05)  # 命令行参数 --min-lr-ratio：余弦调度最低学习率与初始学习率之比；此入口默认 0.05
    p.add_argument("--ema-decay", type=float, default=0.98, help="0 disables EMA")  # 命令行参数 --ema-decay：EMA参数滑动平均衰减系数，0表示完全禁用EMA；此入口默认 0.98
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables gradient clipping")  # 命令行参数 --grad-clip：梯度总范数裁剪阈值，0禁用；此入口默认 1.0
    p.add_argument("--aux-weight", type=float, default=0.1)  # 命令行参数 --aux-weight：两路辅助坐标预测损失的系数，0关闭辅助监督；此入口默认 0.1
    p.add_argument("--distance-weight", type=float, default=0.1)  # 命令行参数 --distance-weight：主预测额外3D欧氏距离损失的系数，0关闭；此入口默认 0.1
    p.add_argument("--stat-weight-file", type=Path,  # 命令行参数 --stat-weight-file：fixed模式的8个统计输入倍率JSON文件路径；此入口默认 EXPERIMENT / "feature_weights" / "paper_selected8.json"
                   default=EXPERIMENT / "feature_weights" / "paper_selected8.json",  # 固定输入倍率默认配置路径；仅fixed训练模式需要读取
                   help="Named input multipliers JSON; used for fixed-mode training only")  # 说明该JSON按特征名称保存倍率，测试时从检查点恢复而非重读它
    p.add_argument("--stat-weighting", choices=("none", "learned", "fixed"), default="fixed",  # 命令行参数 --stat-weighting：none不加权；learned从1学习倍率；fixed从JSON加载固定倍率；此入口默认 "fixed"
                   help="none: identity; learned: trainable; fixed: previously learned input multipliers")  # 说明三种输入加权模式的含义，默认值仍由上一行指定
    p.add_argument("--feature-noise", type=float, default=0.0,  # 命令行参数 --feature-noise：训练时对标准化统计特征和有效路径时延加入的高斯噪声标准差；0关闭；此入口默认 0.0
                   help="Optional std of training-only noise on standardized stats/delay")  # 噪声只在训练批次加入统计输入和物理时延，验证/测试不加入
    p.add_argument("--selection-metric", choices=("coordinate_mse", "mean_error_3d"), default="coordinate_mse")  # 命令行参数 --selection-metric：验证集最佳模型选择指标，越小越好；不使用测试误差选轮；此入口默认 "coordinate_mse"
    return p  # 返回解析器供解析与参数校验复用


def validate_args(a, p):  # 对解析后的训练、网络、损失和设备相关参数做范围检查
    if min(a.epochs, a.batch_size, a.cnn_width, a.d_model, a.num_heads, a.ffn_ratio) < 1:  # 训练轮数、批大小及主要网络维度必须至少为1
        p.error("epochs, batch size and model dimensions must be positive")  # 使用命令行错误格式报告不合法正数参数
    if min(a.cnn_blocks, a.num_layers, a.patience, a.num_workers, a.warmup_epochs) < 0:  # 块数、层数、耐心、工作进程和预热轮数均不能为负
        p.error("Counts must be nonnegative")  # 负计数参数直接终止
    if not 1 <= a.max_paths <= 15 or a.d_model % a.num_heads:  # 路径上限不超过15，且模型宽度必须能被注意力头数整除
        p.error("max-paths must be 1..15; d-model must be divisible by num-heads")  # 报告路径数或MHSA维度约束错误
    if not all(math.isfinite(v) for v in (a.lr, a.weight_decay, a.dropout, a.ema_decay,  # 检查浮点超参数时，先收集优化器、dropout及EMA参数
                                         a.grad_clip, a.aux_weight, a.distance_weight,  # 继续收集梯度裁剪及辅助/距离损失系数
                                         a.feature_noise, a.min_lr_ratio)):  # 再检查训练噪声与最低学习率比例，所有参数都必须有限
        p.error("Floating-point options must be finite")  # 拒绝NaN或无穷超参数
    if a.lr <= 0 or min(a.weight_decay, a.grad_clip, a.aux_weight, a.distance_weight, a.feature_noise) < 0:  # 学习率必须大于零，其余这些系数可以为零但不能为负
        p.error("Invalid optimizer, loss or noise values")  # 优化器、损失或噪声系数不合法时停止
    if not 0 <= a.dropout < 1 or not 0 <= a.ema_decay < 1 or not 0 <= a.min_lr_ratio <= 1:  # 检查丢弃概率、EMA衰减与最低学习率比例的取值范围
        p.error("Invalid dropout, EMA or learning-rate ratio")  # 报告比例参数越界
    if a.mode == "test" and (a.checkpoint is None or a.skip_test):  # 显式测试必须指定检查点，且不能同时要求跳过测试
        p.error("Test requires --checkpoint and cannot use --skip-test")  # 检查测试入口参数是否自洽
    if a.mode == "train" and a.checkpoint:  # 本实现训练不支持断点续训，checkpoint仅给测试使用
        p.error("Training resume is not implemented; --checkpoint is for final test")  # 防止用户误以为提供checkpoint就会接着训练
    return a  # 返回已验证的参数对象


def parse_args(argv=None):  # 提供可传入参数列表的统一解析入口，供run.py和train.py复用
    parser = build_parser()  # 创建完整参数解析器
    return validate_args(parser.parse_args(argv), parser)  # 先解析，再做范围和模式检查


def learning_rate(epoch, args):  # 计算当前轮学习率：可选固定学习率或预热加余弦衰减
    if args.scheduler == "none":  # 关闭调度时，每轮都使用初始学习率
        return args.lr  # 返回命令行设定的固定学习率
    warmup = min(args.warmup_epochs, max(args.epochs - 1, 0))  # 预热轮数不超过总轮数减1，保留衰减阶段
    if epoch <= warmup:  # 前warmup轮采用线性预热；warmup为0时不会进入该分支
        return args.lr * epoch / warmup  # 从较低学习率逐轮线性增加到初始学习率
    progress = (epoch - warmup - 1) / max(args.epochs - warmup - 1, 1)  # 将预热后的轮数映射为余弦衰减进度，分母至少为1
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2)  # 学习率从初始值逐渐降至初始值乘min_lr_ratio


def main(argv=None):  # 单次训练或单次测试的完整执行入口
    args = parse_args(argv)  # 解析并验证此次运行实际使用的命令行参数
    try:  # 将PyTorch依赖导入失败转换为更清楚的环境提示
        import torch  # 导入PyTorch张量计算、自动求导及CUDA接口
        from torch.utils.data import DataLoader, TensorDataset  # 把两路输入、掩码和标签组织成数据集与批次加载器
    except ImportError as error:  # 捕获未安装PyTorch或依赖导入失败的情况
        raise SystemExit("Activate your existing NumPy/PyTorch environment.") from error  # 提示切换到已有训练环境并终止
    sys.path.insert(0, str(EXPERIMENT))  # 优先从当前run.py所在目录加载配套model.py
    from model import SDFNet  # 导入当前脚本目录中的SDFNet网络定义

    random.seed(args.seed)  # 固定Python标准随机数生成器的种子
    np.random.seed(args.seed)  # 固定NumPy随机种子
    torch.manual_seed(args.seed)  # 固定PyTorch CPU及相关随机初始化的种子
    if torch.cuda.is_available():  # 仅在CUDA可用时设置GPU相关随机种子
        torch.cuda.manual_seed_all(args.seed)  # 为所有可见CUDA设备设置种子
    torch.backends.cudnn.benchmark = False  # 不根据输入动态搜索卷积算法，以减少运行差异
    torch.backends.cudnn.deterministic = True  # 要求cuDNN尽量使用确定性实现；不同平台/版本仍可能存在差异
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)  # auto优先CUDA，否则使用用户指定的设备
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):  # 限定CPU/CUDA，并拒绝在无GPU环境强制CUDA
        raise ValueError("Choose an available CPU or CUDA device")  # 设备不可用时明确报错
    active_splits = ("test",) if args.mode == "test" else ("train", "val")  # 训练仅取训练/验证数组；显式测试仅取测试数组
    splits, normalization = read_prepared(args.data, active_splits)  # 读取对应样本与NPZ保存的物理标准化参数
    data_hash = sha256(args.data)  # 记录本次完整NPZ文件的哈希，便于检查点与数据绑定
    hashes = {name: sha256(EXPERIMENT / name) for name in SOURCE_FILES}  # 记录当前run.py与model.py的源码哈希；添加注释也会改变该哈希
    checkpoint = None  # 默认从头训练，尚未加载历史网络参数
    weight_input_snapshot = None  # 默认没有需要保存的外部固定倍率文件字节
    if args.mode == "test":  # 显式测试分支从指定检查点恢复模型
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)  # 先把检查点安全加载到CPU，随后模型再移到指定设备
        if checkpoint["architecture"] != ARCHITECTURE or checkpoint["data_sha256"] != data_hash:  # 要求架构版本与完整数据文件都匹配训练时记录
            raise ValueError("Checkpoint architecture or data differs")  # 防止使用不同输入数据或不同架构直接测试
        if checkpoint["source_hashes"] != hashes:  # 检查当前两份源码与训练检查点记录是否完全一致
            raise ValueError("Code differs: use the saved run.py/model.py snapshot to test")  # 历史检查点应使用对应运行目录的源码快照测试
        input_config, model_config = checkpoint["input_config"], checkpoint["model_config"]  # 测试的输入处理与网络尺寸以检查点为准，不采用当前默认值重建另一模型
    else:  # 从头训练分支：根据本次命令行创建输入配置和网络配置
        input_config = dict(normalization, max_paths=args.max_paths, angle_encoding=args.angle_encoding)  # 从NPZ标准化参数和此次命令行配置建立输入处理规则
        model_config = dict(physical_dim=5 if args.angle_encoding == "circular" else 3,  # 圆周角度编码使用5通道物理输入，标量编码使用3通道
                            max_paths=args.max_paths, cnn_width=args.cnn_width, cnn_blocks=args.cnn_blocks,  # 传递路径上限、两流CNN宽度和残差块数
                            d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers,  # 传递共享Transformer的token宽度、头数与层数
                            ffn_ratio=args.ffn_ratio, dropout=args.dropout, fusion=args.fusion,  # 传递前馈层宽度倍率、dropout和融合模式
                            pooling=args.pooling, branch=args.branch, shortcut=not args.no_shortcut,  # 传递池化方式、双流/消融设置和早期预测捷径开关
                            coordinate_attention=False,  # 把关闭CA的命令行开关转换成启用CA的模型参数
                            stat_weighting=args.stat_weighting)  # 传入统计输入通道的none/learned/fixed模式
        if args.stat_weighting == "fixed":  # 只有固定输入倍率模式会读取外部权重JSON
            values, provenance, weight_input_snapshot = load_stat_weight_file(args.stat_weight_file)  # 校验倍率并保留来源及原始字节快照
            model_config.update(stat_feature_weights=values, stat_weight_source=provenance)  # 把倍率及来源传给模型；固定倍率作为buffer而非可优化参数
    packed = {name: pack_input(split, input_config) for name, split in splits.items()}  # 将每个已读取集合转换成网络所需的两流张量布局
    datasets = {name: TensorDataset(  # 把每个集合封装成可按样本索引访问的数据集
        torch.from_numpy(packed[name]["statistical"]), torch.from_numpy(packed[name]["physical"]),  # 每个样本的前两项依次为统计网格与物理路径
        torch.from_numpy(packed[name]["path_mask"]), torch.from_numpy(split["target"].astype(np.float32)))  # 后两项依次为布尔路径掩码和独立三维监督标签
        for name, split in splits.items()}  # 为训练/验证或测试分别创建TensorDataset
    generator = torch.Generator().manual_seed(args.seed)  # 单独固定训练DataLoader打乱样本的随机生成器
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,  # 评估用加载器不打乱样本顺序，便于预测与编号对齐
                               num_workers=args.num_workers, pin_memory=device.type == "cuda")  # 按配置加载数据；CUDA时使用锁页内存辅助主机到GPU传输
               for name, ds in datasets.items()}  # 为各个集合建立评估加载器
    model = SDFNet(**model_config).to(device)  # 实例化model.py中的双流网络并移到CPU或CUDA
    if args.mode == "train":  # 训练模式拟合标签变换并执行优化；测试模式仅加载已训练参数
        center, scale = fit_target_transform(splits["train"]["target"])  # 标签中心与尺度只由训练集坐标计算
        model.set_target_transform(center, scale)  # 把训练集标签变换保存到模型buffer供前向反变换使用
    else:  # 测试分支：从检查点恢复模型参数及训练时保存的目标变换
        model.load_state_dict(checkpoint["model_state"], strict=True)  # 恢复检查点所有参数与buffer，要求键名和形状完全匹配
    parameters = sum(p.numel() for p in model.parameters())  # 统计网络可训练参数对象中的元素总数

    def evaluate(network, name):  # 对指定集合执行无梯度评估，返回预测、标签、门控权重和指标
        network.eval()  # 切换评估模式，关闭dropout随机丢弃
        predictions, targets, gates = [], [], []  # 建立逐批收集预测、真实标签和两路门控权重的容器
        with torch.no_grad():  # 评估或EMA更新时不构建梯度图，以节省内存且不反向传播
            for s, p, mask, y in loaders[name]:  # 从指定集合依次读取统计、物理、掩码和标签批次
                result = network(s.to(device), p.to(device), mask.to(device), return_aux=True)  # 标签不输入网络；同时获取主预测、辅助预测和门控权重
                predictions.append(result["prediction"].cpu().numpy())  # 把当前批次最终坐标预测移回CPU保存
                gates.append(result["gate"].cpu().numpy())  # 保存当前批次两个分支的动态门控权重
                targets.append(y.numpy())  # 保存按相同顺序读取的真实坐标
        pred, target = np.concatenate(predictions), np.concatenate(targets)  # 沿样本轴合并全部批次
        return pred, target, np.concatenate(gates), metrics(pred, target)  # 返回完整预测、标签、权重以及原坐标单位下的指标

    output = args.output or EXPERIMENT / "runs" / (args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))  # 优先用指定输出路径，否则创建带模式和时间戳的新运行目录
    output.mkdir(parents=True, exist_ok=False)  # 输出目录必须不存在，避免覆盖旧实验
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}  # 把实际命令行参数整理为JSON可写字典，路径转换成字符串
    config.update(architecture=ARCHITECTURE, output=str(output.resolve()), device=str(device),  # 在配置中记录架构标识、绝对输出路径和实际设备
                  data_sha256=data_hash, source_hashes=hashes, parameters=parameters,  # 记录数据哈希、源码哈希和参数数量
                  model_config=model_config, input_config=input_config,  # 保存重建网络与输入整理所需的完整配置
                  statistical_features=STAT_COLUMNS,  # 同时保存特征顺序，防止只看数值时错配
                  physical_features=(["delay_z", "sin_AoA", "cos_AoA", "sin_EoA", "cos_EoA"]  # 圆周模式记录5个物理输入通道的名称
                                     if input_config["angle_encoding"] == "circular" else ["AoA_z", "delay_z", "EoA_z"]),  # 标量模式记录3个标准化通道的实际顺序
                  target_center=model.target_center.cpu().tolist(), target_scale=float(model.target_scale.item()),  # 保存训练标签中心及三个坐标轴共用的缩放值
                  split_ids={name: split["array_id"].tolist() for name, split in splits.items()},  # 记录此次实际读取集合的样本编号，便于核查划分一致性
                  torch_version=str(torch.__version__), numpy_version=np.__version__,  # 保存PyTorch与NumPy版本以辅助复现
                  input_shapes={key: list(value.shape[1:]) for key, value in next(iter(packed.values())).items()},  # 保存单样本各输入形状，去掉样本数量这一维
                  selection="validation only; {}".format(args.selection_metric),  # 说明最佳模型的选择依据是验证集指定指标
                  imputation="existing prepared training means; finite extremes unchanged",  # 记录沿用既有预处理；本脚本不重新删样本或处理有限极值
                  path_order="strongest K if needed, then ascending delay; no power input")  # 记录先按需选最强K条再按时延排序；路径功率不直接输入模型
    if checkpoint is not None:  # 测试模式额外记录所用历史检查点的选择信息
        config["checkpoint_selection"] = checkpoint["selection_metric"]  # 保存最佳检查点采用的验证指标名称
        config["checkpoint_weights"] = checkpoint["weights"]  # 保存历史检查点是原始网络权重还是EMA权重
    save_json(output / "config.json", config)  # 保存当前运行配置；训练结束后还会补充最佳轮数与耗时
    if weight_input_snapshot is not None:  # 固定输入倍率训练时保留外部JSON的原始字节
        (output / "input_feature_weights.json").write_bytes(weight_input_snapshot)  # 写出本次确切使用的固定倍率配置快照
    for name in ("model.py", "run.py", "search.py", "check.py", "README.md"):  # 把当前目录存在的源码与说明复制到新运行目录用于追溯
        if (EXPERIMENT / name).is_file():  # 仅复制确实存在的文件
            shutil.copy2(EXPERIMENT / name, output / name)  # 保存文件快照；复制search/check不会执行这些脚本
    if args.data.with_suffix(".json").is_file():  # 若数据旁边存在同名JSON，保存一份预处理元数据快照
        shutil.copy2(args.data.with_suffix(".json"), output / "preprocessing.json")  # 复制配套JSON；统计列实际索引仍由本脚本常量决定
    print("Device: {} | parameters: {:,} | inputs: {} | output: {}".format(device, parameters, config["input_shapes"], output), flush=True)  # 即时打印实际设备、参数数量、输入形状与保存目录

    if args.mode == "train":  # 训练模式拟合标签变换并执行优化；测试模式仅加载已训练参数
        train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,  # 训练专用加载器每轮打乱训练样本，评估加载器仍保持顺序
                                  generator=generator, num_workers=args.num_workers,  # 使用已固定种子的生成器，并按配置设置加载工作进程
                                  pin_memory=device.type == "cuda")  # CUDA训练时使用锁页内存加载批次
        optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam  # 根据命令行选择AdamW或Adam优化器
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 优化网络可训练参数，固定输入倍率buffer不在其中
        ema = copy.deepcopy(model).eval() if args.ema_decay else None  # EMA系数非零时创建独立的指数滑动平均模型，否则完全关闭EMA
        if ema is not None:  # 仅在启用EMA时更新或冻结EMA副本
            ema.requires_grad_(False)  # EMA参数不通过反向传播优化，只通过滑动平均更新
        best, stale, best_epoch = float("inf"), 0, 0  # 初始化最佳验证分数、连续未改善轮数及最佳轮数
        start = time.perf_counter()  # 记录训练计时起点
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:  # 创建按轮记录训练与验证指标的日志CSV
            writer = csv.writer(handle)  # 创建CSV写入器
            writer.writerow(["epoch", "train_objective", "train_coordinate_mse", "val_coordinate_mse",  # 日志前几列记录轮数、总训练目标、训练坐标MSE和验证坐标MSE
                             "val_mean_error_3d", "lr", "selection_value", "weights"])  # 后几列记录验证3D误差、学习率、模型选择分数和权重类型
            for epoch in range(1, args.epochs + 1):  # 从第1轮开始训练，最多达到epochs，可能因早停提前结束
                lr = learning_rate(epoch, args)  # 按当前轮数计算预热/余弦调度后的学习率
                for group in optimizer.param_groups:  # 遍历优化器各参数组
                    group["lr"] = lr  # 将本轮学习率写入参数组
                model.train()  # 切换训练模式，启用配置中的dropout
                total_loss, total_mse, count = 0.0, 0.0, 0  # 初始化本轮按样本加权的总目标、坐标MSE和样本计数
                for s, p, mask, y in train_loader:  # 依次读取一个训练批次：统计、物理、有效路径标记、真实坐标
                    s, p, mask, y = s.to(device), p.to(device), mask.to(device), y.to(device)  # 把输入、掩码和监督标签都移到训练设备
                    if args.feature_noise:  # 只有训练噪声系数非零时才执行以下增强；验证/测试不进入此处
                        s = s + torch.randn_like(s) * args.feature_noise  # 向已标准化统计输入加入指定标准差的零均值高斯噪声
                        p = p.clone()  # 克隆物理输入后再加噪声，避免就地修改原批次张量
                        delay_column = 0 if input_config["angle_encoding"] == "circular" else 1  # 圆周模式时延在第0列，标量模式时延在第1列
                        p[:, :, delay_column] += torch.randn_like(p[:, :, delay_column]) * args.feature_noise * mask  # 只对真实路径的标准化时延加噪声，不扰动角度编码和填充路径
                    optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度，防止不同批次的梯度意外累积
                    result = model(s, p, mask, return_aux=True)  # 双流前向计算，获得最终坐标、两路辅助坐标及分支有效性
                    delta = (result["prediction"] - y) / model.target_scale  # 将原单位坐标误差除以训练集拟合的共享尺度，稳定优化量级
                    loss = delta.square().mean()  # 主损失为所有样本和三个坐标轴上的归一化MSE
                    if args.distance_weight:  # 可选增加直接约束3D距离的损失项
                        loss = loss + args.distance_weight * (delta.square().sum(1) + 1e-8).sqrt().mean()  # 加入归一化3D欧氏距离均值；根号内的小常数提高数值稳定性
                    if args.aux_weight:  # 辅助系数非零时，让两流早期特征分别接受坐标监督
                        auxiliary = []  # 收集实际有效分支的辅助坐标MSE
                        for key, available in (("stat_prediction", "stat_available"),  # 先处理统计流辅助预测及其可用样本标记
                                               ("physical_prediction", "physical_available")):  # 再处理物理流辅助预测及其可用样本标记
                            valid = result[available]  # 取出当前分支在哪些样本中真实可用
                            if valid.any():  # 只在至少一个样本具备该分支时计算辅助损失
                                auxiliary.append(((result[key][valid] - y[valid]) / model.target_scale).square().mean())  # 对当前分支有效样本计算共享尺度下的辅助坐标MSE
                        if auxiliary:  # 至少有一路辅助损失时才聚合
                            loss = loss + args.aux_weight * torch.stack(auxiliary).mean()  # 将各有效分支辅助MSE取平均后乘系数，加到主目标
                    if not torch.isfinite(loss).item():  # 检查总训练目标是否出现NaN/无穷值
                        raise FloatingPointError("Nonfinite objective at epoch {}".format(epoch))  # 发现非有限损失立即停止，避免保存无效网络
                    loss.backward()  # 沿两路CNN、CA、共享Transformer、门控与预测头反向计算梯度
                    if args.grad_clip:  # 梯度裁剪阈值非零时限制总梯度范数
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)  # 按全局范数裁剪参数梯度；非有限梯度会报错
                    optimizer.step()  # 使用当前梯度更新可训练参数，一批次完成一次优化步骤
                    if ema is not None:  # 仅在启用EMA时更新或冻结EMA副本
                        with torch.no_grad():  # 评估或EMA更新时不构建梯度图，以节省内存且不反向传播
                            for ep, mp in zip(ema.parameters(), model.parameters()):  # 一一对应EMA参数与刚更新的原始模型参数
                                ep.mul_(args.ema_decay).add_(mp, alpha=1 - args.ema_decay)  # EMA参数 = 衰减率×旧EMA参数 + (1-衰减率)×当前参数
                            for eb, mb in zip(ema.buffers(), model.buffers()):  # 对应两模型的buffer，如目标变换和固定输入倍率
                                eb.copy_(mb)  # buffer直接同步，不对它们进行参数式滑动平均
                    total_loss += loss.item() * len(y)  # 按当前批次样本数累积总训练目标，用于整轮平均
                    total_mse += (result["prediction"].detach() - y).square().mean().item() * len(y)  # 累积原坐标单位的主预测MSE，与归一化总目标区分
                    count += len(y)  # 累加本轮已经处理的训练样本数
                evaluated = ema if ema is not None else model  # 验证与检查点选择使用EMA模型或原始模型中的实际配置版本
                _, _, _, validation = evaluate(evaluated, "val")  # 在固定验证集上关闭梯度评估，不用测试集选择最佳轮
                score = validation[args.selection_metric]  # 取指定验证指标作为最佳模型比较分数，默认坐标MSE
                weight_type = "ema" if ema is not None else "raw"  # 标记本轮候选检查点对应EMA权重还是原始权重
                writer.writerow([epoch, total_loss / count, total_mse / count, validation["coordinate_mse"],  # 写出本轮总目标、训练坐标MSE及验证坐标MSE
                                 validation["mean_error_3d"], lr, score, weight_type])  # 同时写出验证3D均值、学习率、选择分数和权重类型
                handle.flush()  # 立即刷新日志到磁盘，方便训练中查看进展
                print("Epoch {:03d} | train MSE {:.6f} | val MSE {:.6f} | val 3D error {:.4f} | lr {:.2g} | {}".format(  # 构造每轮控制台训练进度信息
                    epoch, total_mse / count, validation["coordinate_mse"], validation["mean_error_3d"], lr, weight_type), flush=True)  # 即时输出当前轮指标，供观察收敛与早停
                if score < best:  # 当前验证指标严格小于历史最佳值时认定改善
                    best, stale, best_epoch = score, 0, epoch  # 更新最佳分数与轮数，并把连续未改善轮数清零
                    torch.save(dict(architecture=ARCHITECTURE, model_state=evaluated.state_dict(),  # 保存最佳模型所有参数和buffer，同时写入架构标识
                                    model_config=model_config, input_config=input_config,  # 保存重建网络与输入整理所需的完整配置
                                    data_sha256=data_hash, source_hashes=hashes, epoch=epoch,  # 绑定训练数据、源码版本及当前轮数
                                    selection_metric=args.selection_metric, selection_value=best,  # 记录最佳模型采用哪个验证指标及对应分数
                                    validation=validation, weights=weight_type), output / "best.pt")  # 写入完整验证指标与权重类型，更新本次运行的best.pt
                else:  # 验证分数未改善的分支，累加连续等待轮数
                    stale += 1  # 验证未改善时，连续等待轮数加1
                if args.patience and stale >= args.patience:  # patience非零且连续未改善达到耐心上限时执行早停；0表示禁用
                    print("Early stopping. Best epoch:", best_epoch, flush=True)  # 打印早停信息及最佳模型所在轮数
                    break  # 退出训练轮次循环，随后恢复最佳检查点
        config.update(training_seconds=time.perf_counter() - start, best_epoch=best_epoch)  # 补充总训练耗时和最佳轮数，标记训练流程已结束
        save_json(output / "config.json", config)  # 保存当前运行配置；训练结束后还会补充最佳轮数与耗时
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)  # 读取本次验证集选择的最佳检查点，而非直接使用最后一轮参数
        model.load_state_dict(checkpoint["model_state"])  # 训练结束恢复验证集最优检查点，而非保留最后一轮参数
        _, _, _, training = evaluate(model, "train")  # 恢复最佳模型后，以评估模式重新计算训练集指标
        pred, target, gates, validation = evaluate(model, "val")  # 计算最佳模型的验证预测、门控权重与验证指标
        selection = dict(architecture=ARCHITECTURE, parameters=parameters, data_sha256=data_hash,  # 构造最佳模型的训练/验证指标记录及版本信息
                         best_epoch=checkpoint["epoch"], best_val_mse=checkpoint["validation"]["coordinate_mse"],  # 记录最佳轮数及其验证坐标MSE
                         selection_metric=checkpoint["selection_metric"], selection_value=checkpoint["selection_value"],  # 记录最佳模型实际的验证选择准则与分数
                         weights=checkpoint["weights"], train=training, validation=validation,  # 同时保存原始/EMA权重标记、训练指标与验证指标
                         units="Source xyz units; thresholds assume metres")  # 数值沿用原始坐标单位；0.5/1/2阈值在坐标为米时才具有米的含义
        export_stat_feature_weights(output, model)  # 导出已恢复检查点中的实际输入倍率；none模式为全1
        save_json(output / "selection_metrics.json", selection)  # 保存训练和验证结果，文件不含测试指标
        export_predictions(output, "validation", splits["val"]["array_id"], pred, target, gates)  # 导出验证集逐样本预测、两流门控及CDF数据
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)  # 把最佳模型的训练/验证指标打印到控制台
        return  # 训练流程在此结束，即使不传--skip-test也不会自动测试
    pred, target, gates, result = evaluate(model, "test")  # 只有显式--mode test才会执行固定测试集推理
    result.update(architecture=ARCHITECTURE, parameters=parameters, data_sha256=data_hash,  # 在测试指标中补充模型架构、参数数目和数据版本
                  best_epoch=checkpoint["epoch"], weights=checkpoint["weights"],  # 标明此次测试使用的检查点轮数与权重类型
                  selection_metric=checkpoint["selection_metric"], selection_value=checkpoint["selection_value"],  # 记录最佳模型实际的验证选择准则与分数
                  units="Source xyz units; thresholds assume metres")  # 数值沿用原始坐标单位；0.5/1/2阈值在坐标为米时才具有米的含义
    export_stat_feature_weights(output, model)  # 导出已恢复检查点中的实际输入倍率；none模式为全1
    save_json(output / "metrics.json", result)  # 保存显式测试集指标，mean_error_3d为测试定位误差均值
    export_predictions(output, "test", splits["test"]["array_id"], pred, target, gates)  # 导出全部测试样本的预测坐标、真实坐标、门控权重和CDF
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)  # 打印测试指标用于检查或汇总


if __name__ == "__main__":  # 仅作为脚本直接执行时启动流程，被导入时不自动训练
    main()  # 进入本文件的命令行执行入口
