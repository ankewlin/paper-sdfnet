"""Synthetic forward/backward and checkpoint checks; no real-data training."""
import argparse
import io
import torch
from model import SingleStreamSDFNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    model = SingleStreamSDFNet().to(device)
    model.set_target_transform([4, 5, 1.5], 2)
    s = torch.randn(3, 8, 4, 4, device=device)
    p = torch.randn(3, 15, 5, device=device)
    mask = torch.arange(15, device=device)[None] < torch.tensor([0, 1, 15], device=device)[:, None]
    model.eval()
    with torch.no_grad():
        pred = model(s, p, mask)
        assert pred.shape == (3, 3) and torch.isfinite(pred).all()
        assert model.fuse_input(s, p, mask).shape == (3, 98, 4, 4)
        torch.testing.assert_close(pred, model(s, p.masked_fill(~mask[:, :, None], float("nan")), mask))
        torch.testing.assert_close(pred[:1], model(s[:1], p[:1], mask[:1]), atol=3e-5, rtol=3e-5)
        assert not hasattr(model, "phys_stem") and not hasattr(model, "gate")
    model.train()
    result = model(s, p, mask, return_aux=True)
    (result["prediction"].square().mean() + result["aux_prediction"].square().mean()).backward()
    for name, value in model.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
    blob = io.BytesIO()
    torch.save(model.state_dict(), blob)
    blob.seek(0)
    loaded = SingleStreamSDFNet().to(device)
    loaded.load_state_dict(torch.load(blob, map_location=device, weights_only=True))
    model.eval()
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(s, p, mask), loaded(s, p, mask))
        for config in (dict(num_layers=0), dict(cnn_blocks=0), dict(shortcut=False),
                       dict(pooling="mean"), dict(max_paths=1), dict(physical_dim=3)):
            other = SingleStreamSDFNet(**config).to(device).eval()
            k, f = config.get("max_paths", 15), config.get("physical_dim", 5)
            assert torch.isfinite(other(s[:1], p[:1, :k, :f], mask[:1, :k])).all()
    print("PASS: early fusion, padding, missing paths, singleton, gradients, checkpoint, ablations.")
    print("Default parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
