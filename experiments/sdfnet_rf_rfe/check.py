"""Small synthetic selector and real input/forward checks; no network training."""
import argparse
import ast

import numpy as np
import torch

from rfe import select_features
from train import HERE, change_feature_constant, import_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=__import__("pathlib").Path, default=HERE.parents[1])
    args = parser.parse_args()
    for file in HERE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8-sig"))
    rng = np.random.default_rng(17)
    x = rng.normal(size=(60, 4, 11))
    x[:, :, -1] = 0  # A zero-information group must be removed.
    y = np.stack((x[:, 0, 0], x[:, 1, 1], x[:, 2, 2]), axis=1)
    result = select_features(x, y, ["f{}".format(i) for i in range(11)])
    assert len(result["selected_features"]) == 8
    assert "f10" not in result["selected_features"]
    assert result["elimination_trace"][0]["removed"] == "f10"
    shared = import_file("shared_check", args.project / "experiments/sdfnet_path_count/_shared.py")
    references, _ = shared.reference_runs(shared.DEFAULT_REFERENCE, [42], shared.DEFAULT_DATA)
    _, config = references[42]
    directory = args.project / "experiments/sdfnet_rf_rfe"
    source = (directory / "run.py").read_text(encoding="utf-8-sig")
    original = import_file("original_check", directory / "run.py")
    selected = original.SOURCE_STAT_COLUMNS[-8:]
    changed = change_feature_constant(source, selected)
    namespace = {"__file__": str(directory / "run.py"), "__name__": "patched_check"}
    exec(compile(changed, str(directory / "run.py"), "exec"), namespace)
    splits, _ = original.read_prepared(shared.DEFAULT_DATA, ("train",))
    split = {k: v[:2] for k, v in splits["train"].items()}
    before = original.pack_input(split, config["input_config"])
    after = namespace["pack_input"](split, config["input_config"])
    indices = [original.SOURCE_STAT_COLUMNS.index(n) for n in selected]
    expected = split["statistical"][:, :, indices].reshape(2, 4, 4, 8).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(after["statistical"], expected)
    np.testing.assert_array_equal(before["physical"], after["physical"])
    np.testing.assert_array_equal(before["path_mask"], after["path_mask"])
    model_module = import_file("model_check", directory / "model.py")
    model = model_module.SDFNet(**config["model_config"]).eval()
    print("Local model parameters:", sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        prediction = model(torch.from_numpy(after["statistical"]),
                           torch.from_numpy(after["physical"]), torch.from_numpy(after["path_mask"]))
    assert prediction.shape == (2, 3) and torch.isfinite(prediction).all()
    print("PASS: synthetic grouped selection; source-only feature change; real input packing; unchanged SAGE; full SDF-Net forward.")


if __name__ == "__main__":
    main()
