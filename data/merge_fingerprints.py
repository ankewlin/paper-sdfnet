"""Merge precomputed fingerprints and SAGE results; run with Python 3.7+.

Each consecutive 16 fingerprint rows form one array. SAGE arr_idx is
one-based within a source file. EoA is elevation: horizontal arrival = 0 deg.
This step preserves raw feature values. Fit imputation on training data later.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("Missing or duplicate headers: {}".format(path))
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError("Malformed CSV rows: {}".format(path))
        return fields, rows


def integer(value, label):
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError("Invalid integer {}: {}".format(label, value))
    return int(number)


def merge(data_dir):
    files = sorted((data_dir / "fp").glob("*_fingerprint.csv"))
    if not files:
        raise ValueError("No fingerprint CSV files found")
    merged, summaries = [], []
    invalid = Counter()
    schema = None
    expected_sage = set()
    coords = ["Rx_pos_x", "Rx_pos_y", "Rx_pos_z"]
    for fp in files:
        source = fp.name[:-len("_fingerprint.csv")]
        sage_path = data_dir / "sage_results" / ("sage_" + source + ".csv")
        expected_sage.add(sage_path.name)
        fp_fields, fp_rows = read_csv(fp)
        sage_fields, sage_rows = read_csv(sage_path)
        required = ["arr_idx", "actual_paths", "actual_iter"] + [
            "p{}_{}".format(i, field)
            for i in range(1, 16)
            for field in ("delay_ns", "AoA_deg", "EoA_deg", "power_dB")
        ]
        if not set(coords).issubset(fp_fields) or not set(required).issubset(sage_fields):
            raise ValueError("Missing required columns: " + source)
        fields = ["array_id", "element_id", "source_id"] + fp_fields + sage_fields
        if len(fields) != len(set(fields)):
            raise ValueError("Conflicting column names: " + source)
        if schema is not None and fields != schema:
            raise ValueError("Inconsistent headers: " + source)
        schema = fields
        if not fp_rows or len(fp_rows) != 16 * len(sage_rows):
            raise ValueError("Expected exactly 16 fingerprint rows per SAGE row: " + source)
        indexed = {}
        for row in sage_rows:
            idx = integer(row["arr_idx"], "arr_idx")
            count = integer(row["actual_paths"], "actual_paths")
            if idx in indexed or not 0 <= count <= 15:
                raise ValueError("Duplicate index or invalid path count: " + source)
            indexed[idx] = row
        if set(indexed) != set(range(1, len(sage_rows) + 1)):
            raise ValueError("SAGE indices must be consecutive from 1: " + source)
        for offset, fp_row in enumerate(fp_rows):
            idx, element = offset // 16 + 1, offset % 16
            sage_row = indexed[idx]
            for col in coords:
                if not math.isfinite(float(fp_row[col])):
                    raise ValueError("Invalid coordinate in {} row {}".format(source, offset + 1))
            # Audit features only; nonexistent path slots are padding, not outliers.
            audit = [(k, fp_row[k]) for k in fp_fields if k not in coords]
            if element == 0:
                for path in range(1, int(float(sage_row["actual_paths"])) + 1):
                    for field in ("delay_ns", "AoA_deg", "EoA_deg", "power_dB"):
                        key = "p{}_{}".format(path, field)
                        audit.append((key, sage_row[key]))
            for col, value in audit:
                try:
                    valid = math.isfinite(float(value))
                except (ValueError, TypeError):
                    valid = False
                if not valid:
                    invalid[col] += 1
            row = dict(array_id=source + ":" + str(idx), element_id=element, source_id=source)
            row.update(fp_row)
            row.update(sage_row)
            merged.append(row)
        summaries.append(dict(source=source, arrays=len(sage_rows), rows=len(fp_rows)))
    extra = {p.name for p in (data_dir / "sage_results").glob("*.csv")} - expected_sage
    if extra:
        raise ValueError("Unmatched SAGE files: " + str(sorted(extra)))
    report = dict(rows=len(merged), arrays=sum(s["arrays"] for s in summaries),
                  columns=len(schema), files=summaries,
                  eoa_convention="Elevation angle in degrees; horizontal arrival = 0",
                  target_rule="xyz from element_id 0; other rows retain element coordinates",
                  split_rule="Split by array_id; never split individual element rows",
                  invalid_feature_counts=dict(invalid),
                  audit_rule="FP per element; SAGE once per array, valid paths only",
                  cleaning="No imputation or finite-outlier filtering in this merge; fit on training split",
                  alignment="User-confirmed consecutive blocks of 16 map to one-based arr_idx")
    return schema, merged, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    output = args.data_dir / "fingerprints_merged.csv"
    report_path = args.data_dir / "merge_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Merge output already exists; preserve or rename it before rerunning")
    fields, rows, report = merge(args.data_dir)
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
