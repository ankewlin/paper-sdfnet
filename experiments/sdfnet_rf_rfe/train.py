"""Select 8 statistical feature types using RF-RFE, then train full SDF-Net.

PowerShell: python experiments/sdfnet_rf_rfe/train.py --device cuda
Dry run verifies the reference and commands without fitting or training.
"""
import argparse
import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def change_feature_constant(source, selected):
    """Insert selected features into a snapshot of the local training source."""
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "STAT_COLUMNS" for t in node.targets)]
    if len(nodes) != 1 or len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("Expected one STAT_COLUMNS assignment and eight unique types.")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ["STAT_COLUMNS = " + repr(selected) + "\n"]
    changed = "".join(lines)
    original_tree, changed_tree = ast.parse(source), ast.parse(changed)
    # Structural equality except that single constant is an explicit guard.
    for tree in (original_tree, changed_tree):
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "STAT_COLUMNS" for t in item.targets):
                item.value = ast.Constant(value=None)
    if ast.dump(original_tree) != ast.dump(changed_tree):
        raise ValueError("Unexpected trainer change beyond feature selection.")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=HERE.parents[1])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--reference-cdf", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-only", action="store_true",
                        help="Save the feature selection and trainer; do not train SDF-Net.")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    shared = import_file("path_count_shared", project / "experiments/sdfnet_path_count/_shared.py")
    if shared.PROJECT.resolve() != project:
        raise ValueError("Project differs from the existing reference launcher.")
    data = (args.data or project / "data/prepared.npz").resolve()
    reference = (args.reference_cdf or shared.DEFAULT_REFERENCE).resolve()
    references, data_hash = shared.reference_runs(reference, args.seeds, data)
    _, config = next(iter(references.values()))
    local_dir = project / "experiments" / "sdfnet_rf_rfe"
    for name in ("run.py", "model.py"):
        if not (local_dir / name).is_file():
            raise FileNotFoundError("Local experiment source missing: " + str(local_dir / name))
    source = (local_dir / "run.py").read_text(encoding="utf-8-sig")
    model_source = (local_dir / "model.py").read_bytes()
    ast.parse(source)
    ast.parse(model_source.decode("utf-8-sig"))
    local_hashes = {name: shared.sha256(local_dir / name) for name in ("run.py", "model.py")}
    trainer = import_file("local_sdfnet_run", local_dir / "run.py")
    splits, normalization = trainer.read_prepared(data, ("train", "val"))
    for _, seed_config in references.values():
        for split in ("train", "val"):
            if splits[split]["array_id"].tolist() != seed_config["split_ids"][split]:
                raise ValueError("Training/validation position IDs differ from reference.")
    output = args.run_root.resolve() / ("rf_rfe_" + shared.stamp())
    snapshot = output / "trainer"
    commands = {seed: shared.train_command(snapshot, cfg, cfg["max_paths"], seed,
                data, args.device, output / ("seed_{}".format(seed)))
                for seed, (_, cfg) in references.items()}
    print("Full SDF-Net; RF-RFE selects 8 of 11 statistical types; all retained SAGE paths.")
    print("Model source: " + str(local_dir / "model.py"))
    print("Trainer source: " + str(local_dir / "run.py"))
    print("Training: {}; validation: {}; seeds: {}".format(
        len(splits["train"]["target"]), len(splits["val"]["target"]), args.seeds))
    print("Output: " + str(output))
    if args.dry_run:
        for command in commands.values():
            print(subprocess.list2cmdline(command))
        print("Dry run passed. No feature selection, network training, or test evaluation.")
        return
    from rfe import select_features
    import sklearn
    selection = select_features(splits["train"]["statistical"],
                                splits["train"]["target"], trainer.SOURCE_STAT_COLUMNS)
    # Preserve the local relative channel order where possible; deterministic
    # append for newly selected types. No ordering is chosen using test results.
    ordering = trainer.STAT_COLUMNS + [n for n in trainer.SOURCE_STAT_COLUMNS
                                      if n not in trainer.STAT_COLUMNS]
    selected = [n for n in ordering if n in selection["selected_features"]]
    selection.update(network_feature_order=selected, sklearn_version=sklearn.__version__,
                     data_sha256=data_hash, training_ids=splits["train"]["array_id"].tolist())
    if {name: shared.sha256(local_dir / name) for name in local_hashes} != local_hashes:
        raise RuntimeError("Local source changed during selection; rerun to snapshot one consistent version.")
    changed = change_feature_constant(source, selected)
    snapshot.mkdir(parents=True, exist_ok=False)
    (snapshot / "run.py").write_text(changed, encoding="utf-8")
    (snapshot / "model.py").write_bytes(model_source)
    shared.write_json(output / "selection.json", selection)
    shutil.copy2(HERE / "rfe.py", output / "rfe.py")
    shutil.copy2(HERE / "train.py", output / "train.py")
    if (HERE / "README.md").is_file():
        shutil.copy2(HERE / "README.md", output / "README.md")
    # Confirm packing uses selected channels and leaves SAGE tensors unchanged.
    import numpy as np
    patched = import_file("selected_sdfnet_run", snapshot / "run.py")
    input_config = dict(normalization, max_paths=config["max_paths"], angle_encoding=config["angle_encoding"])
    small = {key: value[:2] for key, value in splits["train"].items()}
    before, after = trainer.pack_input(small, input_config), patched.pack_input(small, input_config)
    np.testing.assert_array_equal(before["physical"], after["physical"])
    np.testing.assert_array_equal(before["path_mask"], after["path_mask"])
    indices = [trainer.SOURCE_STAT_COLUMNS.index(n) for n in selected]
    expected = small["statistical"][:, :, indices].reshape(2, 4, 4, 8).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(after["statistical"], expected)
    generated_hashes = {name: shared.sha256(snapshot / name) for name in ("run.py", "model.py")}
    manifest = dict(method="Grouped RF-RFE + full SDF-Net", complete=False,
        data=str(data), data_sha256=data_hash, seeds=args.seeds,
        reference_cdf=str(reference), reference_runs={str(s): str(r) for s, (r, _) in references.items()},
        hyperparameters=shared.signature(config), reference_source_hashes=config["source_hashes"],
        local_source_dir=str(local_dir), local_source_hashes=local_hashes,
        reference_parameters=config["parameters"], trained_parameters={},
        source_hashes=generated_hashes, selected_features=selected,
        commands={str(s): command for s, command in commands.items()},
        selection="Training-only RF-RFE. SDF-Net checkpoint selected by original validation criterion.",
        evaluation="No test filtering or evaluation during training; saved run.py supports explicit --mode test.")
    shared.write_json(output / "study.json", manifest)
    print("Selected features: " + ", ".join(selected), flush=True)
    if args.select_only:
        print("Selection saved. To train these exact saved features, execute the commands in study.json.")
        return
    for seed, command in commands.items():
        print("Training full SDF-Net with RF-RFE inputs; seed={}".format(seed), flush=True)
        shared.execute(command)
        saved = shared.check_run(output / ("seed_{}".format(seed)), seed, config["max_paths"], data_hash)
        if shared.signature(saved) != shared.signature(config):
            raise ValueError("Training hyperparameters changed.")
        if saved["model_config"] != config["model_config"]:
            raise ValueError("Model configuration differs from the requested reference settings.")
        manifest["trained_parameters"][str(seed)] = saved["parameters"]
        if saved["source_hashes"] != generated_hashes or saved["statistical_features"] != selected:
            raise ValueError("Training source or features changed.")
        shutil.copy2(output / "selection.json", output / ("seed_{}".format(seed)) / "selection.json")
    manifest["complete"] = True
    shared.write_json(output / "study.json", manifest)
    print("Training finished: " + str(output), flush=True)
    print("Run evaluate.py --study \"{}\" --device {} for full test predictions.".format(output, args.device))


if __name__ == "__main__":
    main()
