"""Synthetic-only model checks. No dataset training, no downloaded dependencies."""
import argparse
import io
from pathlib import Path
from run import load_stat_weight_file
import torch
from model import SDFNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stat-weighting", choices=("none", "learned", "fixed"), default="fixed")
    parser.add_argument("--stat-weight-file", type=Path, default=Path(__file__).resolve().parent / "feature_weights/paper_selected8.json")
    args = parser.parse_args()
    weight_options = dict(stat_weighting=args.stat_weighting)
    if args.stat_weighting == "fixed":
        values, provenance, _ = load_stat_weight_file(args.stat_weight_file)
        weight_options.update(stat_feature_weights=values, stat_weight_source=provenance)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = torch.device(args.device)
    s = torch.randn(3, 8, 4, 4, device=device)
    p = torch.randn(3, 15, 5, device=device)
    mask = torch.arange(15, device=device)[None, :] < torch.tensor([0, 1, 15], device=device)[:, None]
    model = SDFNet(**weight_options).to(device).eval()
    model.set_target_transform([4, 5, 1.5], 2.0)
    with torch.no_grad():
        result = model(s, p, mask, return_aux=True)
        assert result["prediction"].shape == (3, 3) and torch.isfinite(result["prediction"]).all()
        assert result["gate"][0, 1] == 0
        changed = p.masked_fill(~mask.unsqueeze(-1), float("nan"))
        torch.testing.assert_close(model(s, changed, mask), result["prediction"])
        # Other samples in a batch must not change a sample's inference result.
        torch.testing.assert_close(model(s[:1], p[:1], mask[:1]), result["prediction"][:1], atol=2e-5, rtol=2e-5)
    model.train()
    result = model(s, p, mask, return_aux=True)
    loss = sum(result[key].square().mean() for key in ("prediction", "stat_prediction", "physical_prediction"))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for prefix in ("stat_stem", "phys_stem", "gate", "layers", "head"):
        assert any(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)), prefix
    model.eval()
    blob = io.BytesIO()
    torch.save(model.state_dict(), blob)
    blob.seek(0)
    reloaded = SDFNet(**weight_options).to(device)
    reloaded.load_state_dict(torch.load(blob, map_location=device, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(s, p, mask), reloaded(s, p, mask))
        for branch in ("stat", "physical"):
            single = SDFNet(branch=branch, **weight_options).to(device).eval()
            before = single(s, p, mask)
            after = (single(s, p * 100, ~mask) if branch == "stat" else single(s * 100, p, mask))
            torch.testing.assert_close(before, after)
            assert torch.isfinite(before).all()
        for options in (dict(num_layers=0), dict(cnn_blocks=0), dict(shortcut=False),
                        dict(fusion="concat", pooling="mean"), dict(max_paths=1), dict(physical_dim=3)):
            other = SDFNet(**options, **weight_options).to(device).eval()
            k, f = options.get("max_paths", 15), options.get("physical_dim", 5)
            assert torch.isfinite(other(s[:1], p[:1, :k, :f], mask[:1, :k])).all()
    print("PASS: finite forward/backward, all-missing paths, padding invariance, singleton batch,")
    print("checkpoint round-trip, branch isolation, and ablation configurations.")
    print("Default parameters: {:,}. No experimental training was performed.".format(sum(p.numel() for p in model.parameters())))


if __name__ == "__main__":
    main()
