"""Quick synthetic checks for this experiment's fixed ablation settings."""
import io

import torch

from model import SDFNet
from run import STAT_COLUMNS, build_parser


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    args = build_parser().parse_args([])
    channels = len(STAT_COLUMNS)
    physical_dim = 5 if args.angle_encoding == "circular" else 3
    model = SDFNet(physical_dim=physical_dim, max_paths=args.max_paths,
                   num_layers=args.num_layers, branch=args.branch).eval()
    stats = torch.randn(3, channels, 4, 4)
    paths = torch.randn(3, args.max_paths, physical_dim)
    mask = torch.arange(args.max_paths)[None, :] < torch.tensor([0, 1, args.max_paths])[:, None]
    with torch.no_grad():
        before = model(stats, paths, mask)
        assert before.shape == (3, 3) and torch.isfinite(before).all()
        padded = paths.masked_fill(~mask.unsqueeze(-1), float("nan"))
        torch.testing.assert_close(before, model(stats, padded, mask))
        if args.branch == "physical":
            torch.testing.assert_close(before, model(stats * 100, paths, mask))
        elif args.branch == "stat":
            torch.testing.assert_close(before, model(stats, paths * 100, ~mask))
        torch.testing.assert_close(before[:1], model(stats[:1], paths[:1], mask[:1]),
                                   atol=2e-5, rtol=2e-5)
    model.train()
    result = model(stats, paths, mask, return_aux=True)
    loss = result["prediction"].square().mean()
    if args.branch != "physical":
        loss = loss + result["stat_prediction"].square().mean()
    if args.branch != "stat":
        loss = loss + result["physical_prediction"][1:].square().mean()
    loss.backward()
    prefixes = ("phys_stem", "head", "layers") if args.branch == "physical" else (
        ("stat_stem", "head", "layers") if args.branch == "stat" else
        ("stat_stem", "phys_stem", "head", "layers"))
    for prefix in prefixes:
        if prefix == "layers" and args.num_layers == 0:
            continue
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                             for g in grads), prefix
    model.eval()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = SDFNet(physical_dim=physical_dim, max_paths=args.max_paths,
                      num_layers=args.num_layers, branch=args.branch).eval()
    restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(model(stats, paths, mask), restored(stats, paths, mask))
    print("PASS: shapes, finite output/gradient, branch isolation, padding, batch and checkpoint")
    print("Parameters: {:,}; no experimental training.".format(sum(p.numel() for p in model.parameters())))


if __name__ == "__main__":
    main()
