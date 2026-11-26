import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import S4DLanguageModel

CONTEXT = 128
BASE_BATCH = 64
EPOCH_TOKENS = 1_048_576
POWERS = {"constant": 0.0, "sqrt": 0.5, "linear": 1.0}
ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_data(path, device):
    text = path.read_bytes().decode("utf-8")
    vocabulary = sorted(set(text))
    index = {c: i for i, c in enumerate(vocabulary)}
    tokens = torch.tensor([index[c] for c in text], dtype=torch.long, device=device)
    a, b = int(0.9 * len(tokens)), int(0.95 * len(tokens))
    return (tokens[:a], tokens[a:b], tokens[b:]), vocabulary


def make_optimizer(model, batch, weight_rule, dynamics_rule, epochs):
    ordinary = [p for p in model.parameters() if not hasattr(p, "_optim")]
    dynamics = [p for p in model.parameters() if hasattr(p, "_optim")]
    k = batch / BASE_BATCH
    optimizer = torch.optim.AdamW([
        {"params": ordinary, "lr": 0.01 * k ** POWERS[weight_rule], "weight_decay": 0.01},
        {"params": dynamics, "lr": 0.001 * k ** POWERS[dynamics_rule], "weight_decay": 0.0},
    ], betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, data, starts=None):
    training = model.training
    model.eval()
    if starts is None:
        starts = torch.arange(0, len(data) - CONTEXT, CONTEXT, device=data.device)
    offsets = torch.arange(CONTEXT, device=data.device)
    total = 0.0
    for part in starts.split(64):
        idx = part[:, None] + offsets
        total += F.cross_entropy(model(data[idx]).flatten(0, 1), data[idx + 1].flatten(), reduction="sum").item()
    model.train(training)
    return total / (len(starts) * CONTEXT)


@torch.no_grad()
def response_profile(model):
    energy = torch.stack([layer.kernel(L=CONTEXT).square().mean(0) for layer in model.s4_layers]).mean(0)
    return (energy / energy.sum().clamp_min(1e-30)).cpu().numpy()


@torch.no_grad()
def prediction_diagnostic(model, data, probe):
    model.eval()
    idx = probe[:8, None] + torch.arange(CONTEXT, device=data.device)
    probabilities = model(data[idx]).softmax(-1).reshape(-1, model.vocab_size)
    return {
        "input_conditioned_probability_rms": (probabilities - probabilities.mean(0)).square().mean().sqrt().item(),
        "unique_top1_predictions": probabilities.argmax(-1).unique().numel(),
    }


def train(args, datasets, vocabulary, probe, seed, batch, weight_rule, dynamics_rule):
    policy = "reference" if batch == BASE_BATCH else f"w{weight_rule}_d{dynamics_rule}"
    folder = args.output / f"b{batch}_{policy}_s{seed}"
    if (folder / "result.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = S4DLanguageModel(len(vocabulary)).to(args.device)
    generator = torch.Generator().manual_seed(seed + 202600)
    starts = torch.randint(len(datasets[0]) - CONTEXT, (args.epochs * EPOCH_TOKENS // CONTEXT,), generator=generator).to(args.device)
    optimizer, scheduler = make_optimizer(model, batch, weight_rule, dynamics_rule, args.epochs)
    config = dict(seed=seed, batch=batch, weight_rule=weight_rule, dynamics_rule=dynamics_rule,
                  epochs=args.epochs, context=CONTEXT, tokens=args.epochs * EPOCH_TOKENS,
                  steps=args.epochs * EPOCH_TOKENS // (batch * CONTEXT),
                  ordinary_peak_lr=optimizer.param_groups[0]["lr"], dynamics_peak_lr=optimizer.param_groups[1]["lr"])
    save_json(folder / "config.json", config)
    profiles = []
    updates = 0

    def record(progress, online_loss):
        profiles.append(response_profile(model))
        row = dict(progress=progress, tokens=updates * batch * CONTEXT, steps=updates,
                   online_train_nll=online_loss, train_nll=evaluate(model, datasets[0], probe),
                   valid_nll=evaluate(model, datasets[1]))
        print(f"{folder.name} {progress}/{args.epochs * 2} train={row['train_nll']:.5f} valid={row['valid_nll']:.5f}", flush=True)
        return row

    offsets = torch.arange(CONTEXT, device=args.device)
    with (folder / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["progress", "tokens", "steps", "online_train_nll", "train_nll", "valid_nll"])
        writer.writeheader()
        writer.writerow(record(0, float("nan")))
        stream.flush()
        windows_per_half = EPOCH_TOKENS // (2 * CONTEXT)
        for progress in range(1, args.epochs * 2 + 1):
            begin, end = (progress - 1) * windows_per_half, progress * windows_per_half
            loss_sum = 0.0
            for current in range(begin, end, batch):
                idx = starts[current:current + batch, None] + offsets
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(datasets[0][idx]).flatten(0, 1), datasets[0][idx + 1].flatten())
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * batch
                updates += 1
            row = record(progress, loss_sum / windows_per_half)
            writer.writerow(row)
            stream.flush()
            if progress % 2 == 0:
                scheduler.step()
    result = dict(status="complete", steps=updates, tokens=updates * batch * CONTEXT,
                  train_nll=row["train_nll"], valid_nll=row["valid_nll"], test_nll=evaluate(model, datasets[2]))
    if seed == 11:
        result.update(prediction_diagnostic(model, datasets[0], probe))
    np.savez(folder / "responses.npz", profiles=np.array(profiles))
    torch.save(dict(model=model.state_dict(), config=config), folder / "final.pt")
    save_json(folder / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/tinyshakespeare.txt")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batches", nargs="+", type=int, choices=[64, 256, 512], default=[64, 256, 512])
    parser.add_argument("--weight-rules", nargs="+", choices=["linear", "sqrt"], default=["linear", "sqrt"])
    parser.add_argument("--dynamics-rules", nargs="+", choices=list(POWERS), default=list(POWERS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    datasets, vocabulary = load_data(args.data, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "data.json", dict(vocabulary=vocabulary, train_characters=len(datasets[0])))
    generator = torch.Generator().manual_seed(2026)
    probe = torch.randint(len(datasets[0]) - CONTEXT, (512,), generator=generator).to(args.device)
    np.save(args.output / "training_probe.npy", probe.cpu().numpy())
    counts = torch.bincount(datasets[0][1:], minlength=len(vocabulary)).double()
    probabilities = counts / counts.sum()
    targets = datasets[0][probe[:, None] + torch.arange(1, CONTEXT + 1, device=args.device)]
    save_json(args.output / "unigram_reference.json", dict(train_probe_nll=-probabilities[targets].log().mean().item()))
    for seed in args.seeds:
        for batch in args.batches:
            policies = [("linear", "linear")] if batch == BASE_BATCH else [(w, d) for w in args.weight_rules for d in args.dynamics_rules]
            for weight_rule, dynamics_rule in policies:
                train(args, datasets, vocabulary, probe, seed, batch, weight_rule, dynamics_rule)


if __name__ == "__main__":
    main()
