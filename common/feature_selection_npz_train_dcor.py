# -*- coding: utf-8 -*-
"""按原评分逻辑，使用 NPZ 的固定训练集全部天线进行特征筛选。

数据流向：
    NPZ + 同名 JSON 列定义
      -> train_indices（不重新划分，不用验证/测试特征拟合）
      -> 反标准化；每个阵列选择功率最大的有效 SAGE 路径
      -> 展开 16 根天线：[主径 3 参数 + 天线 11 统计量]
      -> 三轴互信息平均 R -> min-max 归一化 Rn
      -> 平均绝对 Pearson 冗余 D -> 贪心评分 J = Rn - D
      -> 累计归一化相关性分数首次达到 95% 时取对应前缀。

      conda activate csi-positioning

python D:\loc_paper_SDF_net\common\feature_selection_npz_train.py --source D:\loc_paper_SDF_net\data\prepared.npz
"""



# 导入命令行参数解析模块
import argparse
# 导入哈希模块，记录输入数据和脚本的 SHA-256 指纹。
import hashlib
# 导入 JSON 模块，读取列定义并保存筛选结果。
import json
# 导入系统接口，获取版本信息并设置终端编码。
import sys
# 导入日期和时区工具，用北京时间生成结果文件名。
from datetime import datetime, timedelta, timezone
# 导入路径工具，处理 Windows 文件路径。
from pathlib import Path
import dcor
# 导入 NumPy，处理特征矩阵、索引和统计运算。
import numpy as np
# 导入 scikit-learn，记录实际使用的库版本。
import sklearn
# 导入基于近邻的连续变量互信息估计函数；内部执行近邻搜索。
from sklearn.feature_selection import mutual_info_regression


# 固定随机种子为 42，用于互信息估计器的重复值扰动。
SEED = 42  # 保留原互信息估计器的随机种子。
# 设置互信息估计的 kNN 近邻数 k=3。
NEIGHBORS = 3  # 保留原互信息估计器的近邻数。
# 设置累计归一化相关性首次达到 95% 时停止。
COVERAGE = 0.95  # 保留原归一化单特征分数的累计停止阈值。
# 固定主径物理特征的名称与输出顺序。
PHYSICAL = ["delay_ns", "AoA_deg", "EoA_deg"]  # 候选物理参数顺序。


# 定义命令行路径检查函数，将输入字符串转换为绝对路径对象。
def absolute_path(value):
    # 记录结果字段 。
    """显式传入的输入、输出路径必须为绝对路径。"""
    # 把命令行字符串转换为 Path 对象。
    path = Path(value)
    # 检查用户传入的路径是否为绝对路径。
    if not path.is_absolute():
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise argparse.ArgumentTypeError("必须传入绝对路径：" + value)
    # 返回通过检查的路径对象。
    return path


# 定义文件哈希函数，便于确认复算使用的具体文件版本。
def sha256(path):
    # 记录结果字段 。
    """同时记录数值文件与列定义文件的哈希，便于追溯。"""
    # 创建一个空的 SHA-256 摘要计算器。
    digest = hashlib.sha256()
    # 以二进制只读方式打开待计算哈希的文件。
    with path.open("rb") as stream:
        # 每次读取 1 MB，读到文件末尾的空字节串时结束。
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            # 将当前数据块加入哈希计算。
            digest.update(block)
    # 返回十六进制形式的文件哈希值。
    return digest.hexdigest()


# 定义训练数据构造函数；只用 NPZ 保存的训练索引构造筛选样本。
def read_training(source, metadata_path):
    # 记录结果字段 。
    """只构造固定训练集的特征矩阵和目标矩阵。"""
    # 读取配套 JSON，兼容带 UTF-8 BOM 的文件。
    with metadata_path.open(encoding="utf-8-sig") as stream:
        # 将 JSON 内容解析为字典。
        metadata = json.load(stream)
    # 从配套 JSON 获取 11 个统计特征名，按该顺序解释 NPZ 的统计列。
    stats = metadata["statistical_columns"]  # 名称跟随当前配套 JSON，不写死筛选答案。
    # 读取 NPZ 物理参数最后一维各列对应的名称。
    physical_columns = metadata["physical_columns"]
    # 开始检查统计特征数量必须为 11 且名称不能重复。
    if (len(stats) != 11 or len(set(stats)) != 11
            # 继续要求每个统计特征名称都是非空字符串。
            or not all(isinstance(name, str) and name for name in stats)):
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("JSON 必须定义 11 个不重复的统计特征名")
    # 要求物理特征列恰好包含时延、AoA、EoA。
    if len(physical_columns) != 3 or set(physical_columns) != set(PHYSICAL):
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("JSON 物理列必须为 delay_ns、AoA_deg、EoA_deg")
    # 将 3 个物理参数名与 11 个统计特征名合成 14 个候选名称。
    names = PHYSICAL + stats
    # 检查合并后的 14 个特征名称是否仍然唯一。
    if len(set(names)) != 14:
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("物理特征名和统计特征名不能重复")

    # 只读加载 NPZ；禁止加载 pickle 对象。
    with np.load(source, allow_pickle=False) as archive:
        # 读取每个阵列的唯一编号并转换为字符串。
        array_ids = archive["array_id"].astype(str)
        # 记录 NPZ 中的总阵列数。
        n = len(array_ids)
        # 要求阵列编号为一维且没有重复。
        if array_ids.ndim != 1 or len(set(array_ids)) != n:
            # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
            raise ValueError("NPZ 阵列编号必须唯一且为一维数组")
        # 读取保存的训练、验证和测试索引，不重新随机划分。
        splits = {key: archive[key + "_indices"] for key in ("train", "val", "test")}
        # 逐个检查三种划分的索引结构。
        for key, indices in splits.items():
            # 要求每个划分的索引均为非空一维数组。
            if (indices.ndim != 1 or not len(indices)
                    # 继续要求划分索引的数据类型为整数。
                    or not np.issubdtype(indices.dtype, np.integer)):
                # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
                raise ValueError("非法划分索引：" + key)
        # 此处只检查索引，不读取验证集或测试集的特征来拟合分数。
        # 要求三个索引集合合并后恰好覆盖 0 到 n-1，以发现重叠、遗漏或越界。
        if not np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(n)):
            # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
            raise ValueError("训练/验证/测试索引有重叠、遗漏或越界")
        # 复制训练索引；后续仅按这些索引提取特征。
        # !!!!!!!!!
        train = splits["train"].copy()
        # 每个训练阵列展开为 16 行，要求总行数大于近邻数。
        if len(train) * 16 <= NEIGHBORS:
            # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
            raise ValueError("训练样本不足以执行当前近邻互信息估计")

        # 读取 11 个统计特征标准化时使用的均值。
        stat_mean = archive["stat_mean"].astype(np.float64)
        # 读取 11 个统计特征标准化时使用的尺度。
        stat_scale = archive["stat_scale"].astype(np.float64)
        # 读取 3 个物理参数标准化时使用的均值。
        phys_mean = archive["phys_mean"].astype(np.float64)
        # 读取 3 个物理参数标准化时使用的尺度。
        phys_scale = archive["phys_scale"].astype(np.float64)
        # 逐项检查统计均值、统计尺度、物理均值和物理尺度。
        for key, value, count in (("stat_mean", stat_mean, 11),
                                  # 本项检查 11 个统计特征的尺度参数。
                                  ("stat_scale", stat_scale, 11),
                                  # 本项检查 3 个物理参数的均值。
                                  ("phys_mean", phys_mean, 3),
                                  # 本项检查 3 个物理参数的尺度。
                                  ("phys_scale", phys_scale, 3)):
            # 检查标准化参数的长度以及是否全部为有限数。
            if value.shape != (count,) or not np.isfinite(value).all():
                # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
                raise ValueError("非法标准化参数：" + key)
            # 对尺度数组额外检查：每个值必须严格大于零。
            if "scale" in key and np.any(value <= 0):
                # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
                raise ValueError("标准差必须大于零：" + key)
            # 要求 JSON 的标准化参数与 NPZ 中的数值完全一致。
            if not np.array_equal(np.asarray(metadata["statistics"][key]), value):
                # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
                raise ValueError("NPZ 与 JSON 的标准化参数不一致：" + key)

        # 只提取 train_indices；恢复原始量纲，保持与原脚本的输入口径接近。
        # 提取训练阵列的统计量，形状为 [训练阵列数,16,11]。
        statistical = archive["statistical"][train].astype(np.float64)
        # 提取训练阵列的多径物理参数，保留路径维度。
        physical = archive["physical"][train].astype(np.float64)
        # 提取有效路径掩码；False 对应填充位置。
        mask = archive["path_mask"][train]
        # 读取各路径功率，用来确定主径。
        power = archive["path_power_db"][train].astype(np.float64)
        # 提取每个训练阵列的三维坐标。
        target = archive["target"][train].astype(np.float64)
        # 记录本次实际使用的训练阵列编号。
        train_ids = array_ids[train].tolist()

    # 记录训练阵列数；这里的 k 是阵列数量，不是互信息近邻数。
    k = len(train)
    # 检查每个阵列确有 16 组统计特征和 1 个三维位置标签。
    if statistical.shape != (k, 16, 11) or target.shape != (k, 3):
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("训练统计量必须为 [阵列数,16,11]，目标必须为 [阵列数,3]")
    # 开始检查物理特征应为三维数组，且阵列数和物理列数正确。
    if (physical.ndim != 3 or physical.shape[0] != k or physical.shape[2] != 3
            # 要求路径掩码和功率数组的尺寸与物理特征的阵列、路径维一致。
            or mask.shape != physical.shape[:2] or power.shape != mask.shape
            # 要求掩码为布尔类型，并且每个阵列至少有一条有效路径。
            or mask.dtype != np.bool_ or not mask.any(axis=1).all()):
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("有效路径数据、功率或掩码的形状不正确，或阵列没有有效路径")
    # 只检查有效路径的功率是否全部为有限数。
    if not np.isfinite(power[mask]).all():
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("有效路径功率含非有限值，不能可靠选择主径")
    # 检查训练坐标是否包含 NaN 或无穷值。
    if not np.isfinite(target).all():
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("训练坐标含非有限值")

    # 按 原始值 = 标准化值 × 尺度 + 均值，恢复统计量原始量纲。
    statistical = statistical * stat_scale + stat_mean
    # 按相同逆变换恢复各路径物理参数的原始量纲。
    physical = physical * phys_scale + phys_mean
    # 无效填充路径可能含 NaN；只有有效路径参与主径选择。
    # 将无效路径功率设为负无穷，再取每个阵列功率最大的有效路径索引。
    main_indices = np.argmax(np.where(mask, power, -np.inf), axis=1)
    # 从每个阵列中取出主径的 3 个物理参数，得到 [k,3]。
    main_physical = physical[np.arange(k), main_indices]
    # 依据 JSON 列定义，找到时延、AoA、EoA 在物理参数中的实际列位置。
    physical_order = [physical_columns.index(name) for name in PHYSICAL]
    # 将主径参数排列为固定的时延、AoA、EoA 顺序。
    main_physical = main_physical[:, physical_order]
    # 把每个阵列的主径参数复制给 16 根天线，开始与统计量拼接。
    features = np.concatenate((np.repeat(main_physical[:, None, :], 16, axis=1),
                               # 沿特征维拼接后展开阵列与天线两维，得到 [k×16,14] 的候选矩阵。
                               statistical), axis=2).reshape(-1, 14)
    # 把每个阵列的目标坐标重复 16 次，与展开后的天线特征逐行对应。
    targets = np.repeat(target, 16, axis=0)  # 明确使用阵列级坐标重复 16 次。
    # 开始记录用于追溯本次数据构造过程的元信息。
    details = {
        # 记录训练阵列的数量。
        "train_arrays": k,
        # 记录展开 16 根天线后的训练行数。
        "training_antenna_rows": len(features),
        # 记录NPZ 保存的原始训练索引。
        "train_indices": train.tolist(),
        # 记录训练阵列编号列表。
        "train_array_ids": train_ids,
        # 记录每个训练阵列的主径索引，编号从 0 开始。
        "main_path_indices_zero_based": main_indices.tolist(),
        # 记录本次训练样本包含的高度。
        "heights": sorted(set(target[:, 2].tolist())),
        # 记录固定划分的使用规则。
        "split_policy": "Use NPZ train_indices in their saved order; no new split",
        # 记录一个阵列的坐标复制给 16 个天线样本的规则。
        "coordinate_policy": "Repeat each NPZ array-level target for its 16 antenna rows",
        # 记录主径的选择和共享规则。
        "physical_policy": "Highest-power valid SAGE path per array, shared by its 16 antennas",
        # 记录NPZ 既有填补值无法反推原始缺失位置的说明。
        "input_imputation_note": "NPZ may already contain imputed values; this run cannot recover their original missing-value mask",
    # 结束当前字典或函数调用中的字典结构。
    }
    # 返回候选特征、坐标标签、特征名称及数据构造记录。
    return features, targets, names, details


# 定义互信息相关性、冗余和贪心筛选的核心过程。
# !!!!!!!!!!!!! ********************
def select_features(features, targets, names):
    # 记录结果字段 。
    """保留原脚本的互信息、Pearson 冗余、贪心及停止逻辑。"""
    # 标记每个特征值是否为有限数，得到与特征矩阵同形状的布尔矩阵。
    # valid异常值处理
    valid = np.isfinite(features)
    # 检查是否存在整列都没有有效数值的特征。
    if np.any(valid.sum(axis=0) == 0):
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("某个特征在训练集中没有有限值")
    # 对每列仅使用有限训练值计算均值，作为缺失值填补值。
    means = np.where(valid, features, 0).sum(axis=0) / valid.sum(axis=0)
    # 用训练列均值填补当前非有限特征值；已有 NPZ 填补值保持原样。
    features = np.where(valid, features, means)  # 仅用训练样本均值补充当前非有限值。

    # 找出非恒定特征的列索引；恒定列不参与近邻估计和 Pearson 计算。
    active = np.flatnonzero(np.any(features != features[0], axis=0)) #找到所有的非恒定列的索引
    """
    features = np.array([
    [1, 2, 3],  # 0（基准）
    [1, 5, 3],  # active = 1 =返回
    [1, 2, 3]   # 2
])
    """
    # 创建互信息矩阵，每行一个特征，每列分别对应 x、y、z。
    mi = np.zeros((len(names), 3))
    # 检查是否所有候选特征都是常数。
    if not len(active):
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("全部候选特征恒定，无法计算当前归一化分数")
    # 依次对 x、y、z 三个位置坐标估计互信息。
    for axis in range(3):
        # 仅在当前坐标有变化时估计互信息；恒定坐标对应互信息保留为 0。
        if np.any(targets[:, axis] != targets[0, axis]):
            # ！！！调用内部包含 kNN 搜索的互信息估计器，计算各非恒定特征与当前坐标的依赖程度。
            mi[active, axis] = mutual_info_regression(
                # 传入全部训练行的有效特征和当前坐标；明确将这些特征视为连续变量。
                features[:, active], targets[:, axis], discrete_features=False,
                # 设置 kNN 的 k=3；种子 42 控制估计器处理重复值时加入的微小随机扰动。
                n_neighbors=NEIGHBORS, random_state=SEED)
    # 对三个坐标互信息求平均，得到论文中的 R_F。
    relevance = mi.mean(axis=1)
    # 计算所有候选相关性分数的最大值与最小值之差。
    span = relevance.max() - relevance.min()
    # 若所有相关性相同，则 min-max 归一化分母为零，停止并报错。
    if span <= 0:
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise ValueError("所有候选相关性分数相同，min-max 归一化无定义")
    # 在完整候选集合上归一化一次，得到 Rn；之后各轮不重复归一化。
    normalized = (relevance - relevance.min()) / span

    values = features[:, active].astype(np.float64, copy=True)
    values /= np.max(np.abs(values), axis=0)

    corr = np.zeros((len(names), len(names)), dtype=np.float64)

    for i, feature_i in enumerate(active):
        corr[feature_i, feature_i] = 1.0

        for j in range(i + 1, len(active)):
            feature_j = active[j]

            value = float(dcor.distance_correlation(
                values[:, i],
                values[:, j],
                exponent=1,
                method="avl",
            ))

            if not np.isfinite(value):
                raise ValueError("距离相关性计算异常：{}、{}".format(
                    names[feature_i], names[feature_j]
                ))

            value = float(np.clip(value, 0.0, 1.0))
            corr[feature_i, feature_j] = value
            corr[feature_j, feature_i] = value

    # 初始化已选索引、尚未选择的非恒定特征索引和逐轮记录。
    selected, remaining, trace = [], active.tolist(), []
    # 计算完整贪心排序；之后再截取首次达到 95% 的前缀作为最终集合。
    while remaining:
        # 对每个剩余候选，计算其与全部已选特征的平均绝对 Pearson 相关性 D。
        penalty = (corr[np.ix_(remaining, selected)].mean(axis=1)
                   # 第一轮已选集合为空，所有候选的冗余惩罚都设为零。
                   if selected else np.zeros(len(remaining)))
        # 计算本轮评分 J=Rn-D；冗余权重为 1。
        scores = normalized[remaining] - penalty
        # 找出当前得分最大的候选在 remaining 中的位置；完全同分时取先出现者。
        pos = int(np.argmax(scores))
        # 取出该候选的原始列索引，并从剩余候选中删除它。
        index = remaining.pop(pos)
        # 将本轮获胜特征的列索引追加到已选集合。
        selected.append(index)
        # 保存本轮入选特征以及添加后的累计相关性占比。
        trace.append({
            # 记录本轮编号、获胜特征名和该特征在候选矩阵中的列索引。
            "k": len(selected), "feature": names[index], "index": index,
            # 记录该特征的原始相关性 R 及归一化相关性 Rn。
            "R": float(relevance[index]), "R_normalized": float(normalized[index]),
            # 记录特征加入前的冗余惩罚 D 和本轮评分 J。
            "D": float(penalty[pos]), "J": float(scores[pos]),
            # 记录已选归一化 R 之和除以全体归一化 R 之和。
            "coverage_normalized": float(normalized[selected].sum() / normalized.sum()),
        # 结束当前字典或函数调用中的字典结构。
        })
    # 寻找累计归一化相关性首次达到 0.95 的迭代编号，作为最终选取数量。
    stop = next(row["k"] for row in trace if row["coverage_normalized"] >= COVERAGE)
    # 截取贪心排序前 stop 个特征；后续排序记录不属于最终选集。
    chosen = [names[index] for index in selected[:stop]]
    # 返回结构化筛选结果。
    return {
        # 记录各特征在本轮筛选中实际填补的数量。
        "imputed_counts_this_run": (~valid).sum(axis=0).tolist(),
        # 记录各特征与三个坐标的互信息。
        "mutual_information_xyz": mi.tolist(), "trace": trace,
        # 记录按停止规则得到的最终子集。
        "stopping_results": {"normalized": {
            # 记录达到阈值时选中的特征数量。
            "count": stop, "selected_features": chosen,
            # 记录停止时的累计归一化相关性占比。
            "coverage": trace[stop - 1]["coverage_normalized"],
        # 结束当前字典或函数调用中的字典结构。
        }},
        # 记录最终子集中的物理特征。
        "selected_physical_features": [name for name in chosen if name in PHYSICAL],
        # 记录最终子集中的统计特征。
        "selected_statistical_features": [name for name in chosen if name not in PHYSICAL],
    # 结束当前字典或函数调用中的字典结构。
    }


# 定义命令行入口，连接数据读取、筛选、结果保存和终端展示。
def main():
    # 明确使用 UTF-8，避免 Windows 重定向输出时中文变成乱码。
    # 依次处理标准输出和标准错误输出流。
    for stream in (sys.stdout, sys.stderr):
        # 检查当前 Python 输出流是否支持重新设置编码。
        if hasattr(stream, "reconfigure"):
            # 将输出设置为 UTF-8，便于保存中文日志。
            stream.reconfigure(encoding="utf-8")
    # 创建命令行解析器，将文件开头说明作为帮助信息。
    parser = argparse.ArgumentParser(description=__doc__)
    # 定义命令行参数及路径检查方式；参数说明由 help 给出。
    parser.add_argument("--source", type=absolute_path, required=True, help="prepared NPZ 绝对路径")
    # 定义命令行参数及路径检查方式；参数说明由 help 给出。
    parser.add_argument("--metadata", type=absolute_path, help="默认使用 NPZ 同目录同名 JSON")
    # 定义命令行参数及路径检查方式；参数说明由 help 给出。
    parser.add_argument("--output", type=absolute_path, help="输出新 JSON；省略时自动生成带时间的文件名")
    # 解析命令行参数；参数不合要求时显示帮助并退出。
    args = parser.parse_args()
    # 解析输入 NPZ 的完整路径。
    source = args.source.resolve()
    # 优先使用指定 JSON，否则使用与 NPZ 同目录同名的 JSON。
    metadata_path = (args.metadata or source.with_suffix(".json")).resolve()
    # 生成北京时间时间戳，精确到微秒，用于避免输出文件重名。
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    # 优先使用指定输出路径；未指定时准备脚本目录下的默认结果目录。
    output = args.output or (Path(__file__).resolve().parent / "feature_selection_runs" /
                             # 将时间戳拼入默认 JSON 文件名。
                             ("npz_train_dcor_" + stamp + ".json"))
    # 检查结果文件是否已经存在，避免覆盖历史记录。
    if output.exists():
        # 发现上述条件不满足时抛出异常，停止本次运行并给出具体原因。
        raise FileExistsError("输出已存在，请换一个文件名：" + str(output))
    # 使用配套 NPZ 和 JSON 构造固定训练集的候选特征矩阵。
    features, targets, names, details = read_training(source, metadata_path)
    # 将当前路径、运行规则或筛选结果打印到终端。
    print("NPZ：", source, flush=True)
    # 将当前路径、运行规则或筛选结果打印到终端。
    print("列定义：", metadata_path, flush=True)
    # 将当前路径、运行规则或筛选结果打印到终端。
    print("训练阵列：{}；天线记录：{}；候选特征：{}".format(
        # 继续填入终端摘要的训练阵列数、天线行数与候选特征数，并立即刷新输出。
        details["train_arrays"], details["training_antenna_rows"], len(names)), flush=True)
    # 将当前路径、运行规则或筛选结果打印到终端。
    print("坐标口径：每个阵列的三维坐标重复给其 16 根天线。", flush=True)
    # 将当前路径、运行规则或筛选结果打印到终端。
    print(    "规则：seed=42，互信息近邻数=3，D为平均距离相关性，"
    "J=Rn-D，累计归一化相关性分数达到95%。", flush=True)
    # 执行互信息和 Pearson 贪心筛选，得到最终子集和完整逐轮记录。
    result = select_features(features, targets, names)
    # 把训练索引、阵列编号及主径选择记录合并到结果中。
    result.update(details)
    # 补充文件指纹、算法参数和运行环境版本，便于以后复现。
    result.update({
        # 记录输入 NPZ 的路径和 SHA-256 指纹。
        "source": str(source), "source_sha256": sha256(source),
        # 记录输入 JSON 的路径和 SHA-256 指纹。
        "metadata": str(metadata_path), "metadata_sha256": sha256(metadata_path),
        # 记录本脚本的 SHA-256 指纹。
        "script_sha256": sha256(Path(__file__).resolve()),
        # 记录完整候选特征名称及固定随机种子、近邻数。
        "feature_names": names, "seed": SEED, "neighbors": NEIGHBORS,
        # 记录累计归一化相关性的停止阈值 0.95。
        "coverage_threshold": COVERAGE,
        # 记录累计分数含义：单特征相关性之和，而不是联合信息。
        "coverage_definition": "Cumulative normalized individual relevance scores; not joint information",
        # 记录记录 Python、NumPy 与 scikit-learn 的版本。
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     # 记录 scikit-learn 版本，便于解释互信息估计的版本差异。
                     "scikit_learn": sklearn.__version__},
    # 结束当前字典或函数调用中的字典结构。
    })
    # 创建结果文件的父目录；目录已存在时继续执行。
    output.parent.mkdir(parents=True, exist_ok=True)
    # 以独占新建模式写入 UTF-8 JSON，禁止覆盖已有文件。
    with output.open("x", encoding="utf-8") as stream:
        # 保存带缩进的 JSON，直接保留中文，并禁止写入 NaN/Infinity。
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    # 读取归一化累计相关性达到 95% 时实际选出的特征数量。
    count = result["stopping_results"]["normalized"]["count"]
    # 依次展示完整排序中的每一步。
    for row in result["trace"]:
        # 把停止步骤以内的特征标为选入，后续特征标为未选。
        mark = "选入" if row["k"] <= count else "未选"
        # 将当前路径、运行规则或筛选结果打印到终端。
        print(("{k:2d} {feature:25s} Rn={R_normalized:.4f} D={D:.4f} "
               # 补充本轮 J 分数和累计比例，格式化后附上选入或未选标记。
               "J={J:.4f} Cnorm={coverage_normalized:.4%}  " + mark).format(**row))
    # 将当前路径、运行规则或筛选结果打印到终端。
    print(json.dumps(result["stopping_results"], ensure_ascii=False, indent=2))
    # 将当前路径、运行规则或筛选结果打印到终端。
    print("结果已保存：", output)


# 仅当直接运行此文件时执行命令行入口；被导入时不自动运行。
if __name__ == "__main__":
    # 启动主流程。
    main()
