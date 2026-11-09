"""Array-level fingerprint preparation, using NumPy and the standard library.

Run: python common/data.py
Import: load_prepared(path) -> split dictionaries containing NumPy arrays.
Coordinates remain in original units. EoA: horizontal arrival is zero degrees.
Only nonnumeric/nonfinite feature values are imputed. Finite outliers are retained.
No path selection, angle encoding or feature selection is performed here.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STAT_COLUMNS = ["Rx_power_dBm", "R_max_dB", "Rise_time_ns", "Tau_MED_ns",
                "Tau_RMS_ns", "Kurtosis", "Skewness", "K_factor", "PL_dB",
                "DPR", "tau_dom_RMS_ns"]
PHYSICAL_COLUMNS = ["delay_ns", "AoA_deg", "EoA_deg"]
COORD_COLUMNS = ["Rx_pos_x", "Rx_pos_y", "Rx_pos_z"]
SPLITS = ("train", "val", "test")


def numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def exact_int(value, name):
    result = numeric(value)
    if not np.isfinite(result) or result != int(result):
        raise ValueError("Invalid {}: {}".format(name, value))
    return int(result)


def read_arrays(path):
    """Return one record per array; duplicated SAGE rows are verified, not learned."""
    groups = defaultdict(list)
    sage_fields = ["actual_paths"] + [
        "p{}_{}".format(i, c) for i in range(1, 16)
        for c in PHYSICAL_COLUMNS + ["power_dB"]]
    required = ["array_id", "element_id"] + COORD_COLUMNS + STAT_COLUMNS + sage_fields
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError("Merged CSV is missing required columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate CSV columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()) or not row["array_id"]:
                raise ValueError("Malformed CSV row")
            groups[row["array_id"]].append(row)
    if not groups:
        raise ValueError("No array samples")
    ids, statistics, physical, masks, labels, powers = [], [], [], [], [], []
    for array_id in sorted(groups):
        rows = sorted(groups[array_id], key=lambda r: exact_int(r["element_id"], "element_id"))
        if len(rows) != 16 or [exact_int(r["element_id"], "element_id") for r in rows] != list(range(16)):
            raise ValueError("Array must have exactly elements 0..15: " + array_id)
        first = rows[0]
        if any(any(r[c] != first[c] for c in sage_fields) for r in rows[1:]):
            raise ValueError("Inconsistent replicated SAGE values: " + array_id)
        coordinates = np.array([[numeric(r[c]) for c in COORD_COLUMNS] for r in rows])
        if not np.isfinite(coordinates).all():
            raise ValueError("Invalid coordinate; labels cannot be mean-imputed: " + array_id)
        if not np.allclose(coordinates[:, 2], coordinates[0, 2], rtol=0, atol=1e-9):
            raise ValueError("Inconsistent height within array: " + array_id)
        count = exact_int(first["actual_paths"], "actual_paths")
        if not 0 <= count <= 15:
            raise ValueError("actual_paths outside 0..15: " + array_id)
        p = np.full((15, 3), np.nan)
        power = np.full(15, np.nan)
        for i in range(count):
            p[i] = [numeric(first["p{}_{}".format(i + 1, c)]) for c in PHYSICAL_COLUMNS]
            power[i] = numeric(first["p{}_power_dB".format(i + 1)])
        ids.append(array_id)
        statistics.append([[numeric(r[c]) for c in STAT_COLUMNS] for r in rows])
        physical.append(p)
        powers.append(power)
        masks.append(np.arange(15) < count)
        labels.append(coordinates[0])
    return dict(array_id=np.array(ids), statistical=np.array(statistics),
                physical=np.array(physical), path_mask=np.array(masks),
                target=np.array(labels), path_power_db=np.array(powers))


def split_by_height(target):
    rng = np.random.default_rng(42)
    split = {name: [] for name in SPLITS}

    for height in np.unique(target[:, 2]):
        indices = rng.permutation(np.flatnonzero(target[:, 2] == height))
        n = len(indices)

        if n < 10:
            raise ValueError("Need at least ten arrays per height split")

        # 每个高度按训练集70%、验证集10%、测试集剩余20%划分。
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        boundaries = (n_train, n_train + n_val)

        for name, part in zip(SPLITS, np.split(indices, boundaries)):
            split[name].extend(part.tolist())

    return {
        name: np.array(sorted(values), dtype=np.int64)
        for name, values in split.items()
    }

def fit_statistics(values, valid_mask):
    """Pool valid training positions per feature, excluding padding and anomalies."""
    flat = values[valid_mask]
    valid = np.isfinite(flat)
    counts = valid.sum(axis=0)
    if (counts == 0).any():
        raise ValueError("A feature has no valid training values; cannot compute its mean")
    mean = np.where(valid, flat, 0).sum(axis=0) / counts
    variance = np.where(valid, (flat - mean) ** 2, 0).sum(axis=0) / counts
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def transform(values, mask, mean, scale):
    result = (np.where(np.isfinite(values), values, mean) - mean) / scale
    result[~mask] = 0
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite values after transformation")
    return result


def prepare(source):
    raw = read_arrays(source)
    split = split_by_height(raw["target"])
    train = split["train"]
    stat_mask = np.ones(raw["statistical"].shape[:2], dtype=bool)
    stat_mean, stat_scale = fit_statistics(raw["statistical"][train], stat_mask[train])
    phys_mean, phys_scale = fit_statistics(raw["physical"][train], raw["path_mask"][train])
    prepared = dict(raw)
    prepared["statistical"] = transform(raw["statistical"], stat_mask, stat_mean, stat_scale)
    prepared["physical"] = transform(raw["physical"], raw["path_mask"], phys_mean, phys_scale)
    prepared["target"] = raw["target"].astype(np.float32)
    prepared["stat_mean"], prepared["stat_scale"] = stat_mean, stat_scale
    prepared["phys_mean"], prepared["phys_scale"] = phys_mean, phys_scale
    report = dict(source=str(Path(source).resolve()),
                  source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  statistical_columns=STAT_COLUMNS, physical_columns=PHYSICAL_COLUMNS,
                  label_columns=COORD_COLUMNS, arrays=len(raw["array_id"]),
                  eoa="Degrees; horizontal arrival = 0", path_order="Original SAGE order",
                  imputation="Training valid mean per feature; finite outliers retained",
                  normalization="Training valid population std per feature; padding excluded",
                  power="Raw auxiliary metadata only; not a model input; padding may be NaN",
                  target="First element xyz in original units, not normalized",
                  splits={})
    for name, indices in split.items():
        prepared[name + "_indices"] = indices
        stat_invalid = (~np.isfinite(raw["statistical"][indices])).sum(axis=(0, 1))
        phys_invalid = ((~np.isfinite(raw["physical"][indices])) & raw["path_mask"][indices, :, None]).sum(axis=(0, 1))
        report["splits"][name] = dict(
            count=len(indices), array_ids=raw["array_id"][indices].tolist(),
            heights={str(h): int((raw["target"][indices, 2] == h).sum()) for h in np.unique(raw["target"][:, 2])},
            statistical_replacements=dict(zip(STAT_COLUMNS, stat_invalid.tolist())),
            physical_replacements=dict(zip(PHYSICAL_COLUMNS, phys_invalid.tolist())))
    report["statistics"] = dict(stat_mean=stat_mean.tolist(), stat_scale=stat_scale.tolist(),
                                phys_mean=phys_mean.tolist(), phys_scale=phys_scale.tolist())
    return prepared, report


def load_prepared(path):
    """Model inputs + labels/IDs per split. Power is auxiliary, never auto-concatenated."""
    with np.load(path, allow_pickle=False) as archive:
        keys = ("array_id", "statistical", "physical", "path_mask", "target", "path_power_db")
        return {name: {k: archive[k][archive[name + "_indices"]] for k in keys} for name in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--source", type=Path, default=root / "fingerprints_merged.csv")
    parser.add_argument("--output", type=Path, default=root / "prepared.npz")
    args = parser.parse_args()
    metadata = args.output.with_suffix(".json")
    if args.output.exists() or metadata.exists():
        raise FileExistsError("Prepared output exists; use a new --output to preserve earlier splits")
    arrays, report = prepare(args.source)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with metadata.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({name: report["splits"][name]["count"] for name in SPLITS}))
    print("Saved {} and {}".format(args.output, metadata))


if __name__ == "__main__":
    main()
