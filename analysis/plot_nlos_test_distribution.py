"""展示现有 CDF 坐标规则如何从全部测试位置筛选 NLoS 评估子集。

运行示例：python analysis/plot_nlos_test_distribution.py
默认读取项目 data/prepared.npz，输出到 analysis/nlos_distribution_时间戳。
规则对应 plot_latest_12_three_seed_cdf.py 中的 nlos_mask。
本脚本展示坐标筛选结果，传播状态的物理判定依据需由场景本身提供。
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def selection_masks(truth):
    """逐步拆分原始规则，保留原代码的严格/非严格边界。"""
    x, y, z = truth.T
    x_region = (
        ((x >= 0.5) & (x <= 3.7))
        | ((x >= 4.7) & (x <= 6.5))
        | ((x >= 7.5) & (x <= 8.4))
    )
    y_region = (y < 2.2) | (y > 7.25)
    region = x_region & y_region
    # 第一条额外排除规则：左下区域中 2.45 < z < 2.55 的位置。
    rule_a = (x >= 0.5) & (x <= 3.7) & (y < 2.2) & (z > 2.45) & (z < 2.55)
    # 第二条额外排除规则：(5, 1, 2.5) 周围指定范围内的位置。
    rule_b = (x > 4.95) & (x < 5.05) & (y > 0.95) & (y < 1.01) & (z > 2.47) & (z < 2.55)
    selected = region & ~(rule_a | rule_b)
    return x_region, y_region, region, rule_a, rule_b, selected


def draw_distribution(truth, selected, output):
    """Plot only the selected positions in 3D and in the xy plane."""
    points = truth[selected]
    if not len(points):
        raise ValueError("No test positions match the selection rules")
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(12, 5.6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    scatter = ax3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=points[:, 2], cmap="viridis", s=48, depthshade=False)
    ax2d.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="viridis",
                 s=48, vmin=points[:, 2].min(), vmax=points[:, 2].max())
    lower = truth.min(axis=0) - 0.3
    upper = truth.max(axis=0) + 0.3
    ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="3D positions",
             xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    ax3d.set_box_aspect(upper - lower)
    ax3d.view_init(elev=25, azim=-55)
    ax2d.set(xlabel="x (m)", ylabel="y (m)", title="Top view",
             xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    ax2d.set_aspect("equal")
    ax2d.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax2d, label="Height z (m)", shrink=0.8, pad=0.04)
    fig.suptitle(f"NLoS test positions (n = {len(points)})", fontsize=15)
    fig.text(0.5, 0.025, "Points at different heights may overlap in the top view.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"nlos_test_distribution.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="默认使用项目 data/prepared.npz")
    parser.add_argument("--output", type=Path, help="可选的全新输出文件夹")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data = (args.data or project / "data" / "prepared.npz").resolve()
    with np.load(data, allow_pickle=False) as archive:
        indices = archive["test_indices"]
        all_truth = archive["target"]
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("test_indices 必须为一维整数数组")
        if not len(indices) or len(np.unique(indices)) != len(indices):
            raise ValueError("测试索引为空或存在重复")
        if np.any(indices < 0) or np.any(indices >= len(all_truth)):
            raise ValueError("测试索引越界")
        ids = np.asarray([str(value) for value in archive["array_id"][indices]])
        truth = all_truth[indices].astype(np.float64)
    if truth.shape != (len(ids), 3) or not np.isfinite(truth).all():
        raise ValueError("测试坐标必须为有限的 [N,3] 数组")
    if len(set(ids)) != len(ids):
        raise ValueError("测试位置编号重复")
    # 按字符串编号排序，与 CDF 脚本的 reference 顺序保持一致。
    order = np.argsort(ids)
    ids, truth, indices = ids[order], truth[order], indices[order]
    masks = selection_masks(truth)
    x_region, y_region, region, rule_a, rule_b, selected = masks
    removed_a = region & rule_a
    removed_b = region & ~rule_a & rule_b
    counts = {
        "all_test": len(ids), "after_x_rule": int(x_region.sum()),
        "after_xy_rule": int(region.sum()), "removed_by_rule_a": int(removed_a.sum()),
        "removed_by_rule_b_after_a": int(removed_b.sum()), "selected": int(selected.sum()),
    }
    output = (args.output or project / "analysis" /
              ("nlos_distribution_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "nlos_points.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["array_id", "npz_row", "x_m", "y_m", "z_m"])
        for i in np.flatnonzero(selected):
            writer.writerow([ids[i], int(indices[i]), *truth[i].tolist()])
    summary = {
        "data": str(data), "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "reference_script": "plot_latest_12_three_seed_cdf.py",
        "meaning": "现有CDF坐标规则定义的评估子集", "counts": counts,
        "rules": {
            "x": "[0.5,3.7] or [4.7,6.5] or [7.5,8.4]",
            "y": "y < 2.2 or y > 7.25",
            "exclude_a": "0.5 <= x <= 3.7 and y < 2.2 and 2.45 < z < 2.55",
            "exclude_b": "4.95 < x < 5.05 and 0.95 < y < 1.01 and 2.47 < z < 2.55",
        },
        "selected_array_ids": ids[selected].tolist(),
        "height_counts": [
            {"z_m": float(z), "all_test": int((truth[:, 2] == z).sum()),
             "selected": int((selected & (truth[:, 2] == z)).sum())}
            for z in np.unique(truth[:, 2])
        ],
    }
    (output / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_distribution(truth, selected, output)
    print("全部测试位置：{}".format(counts["all_test"]))
    print("按x范围筛选后：{}".format(counts["after_x_rule"]))
    print("再按y边缘区域筛选后：{}".format(counts["after_xy_rule"]))
    print("额外规则A剔除：{}；规则B再剔除：{}".format(
        counts["removed_by_rule_a"], counts["removed_by_rule_b_after_a"]))
    print("最终保留：{}".format(counts["selected"]))
    print("输出目录：{}".format(output))


if __name__ == "__main__":
    main()
