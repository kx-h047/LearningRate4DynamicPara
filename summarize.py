import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.results.glob("b*")):
        if not (folder / "result.json").exists():
            continue
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "result.json").read_text())
        reference = args.results / f"b64_reference_s{config['seed']}"
        history = np.genfromtxt(folder / "training.csv", delimiter=",", names=True)
        baseline = np.genfromtxt(reference / "training.csv", delimiter=",", names=True)
        profiles = np.load(folder / "responses.npz")["profiles"]
        reference_profiles = np.load(reference / "responses.npz")["profiles"]
        rows.append(dict(batch=config["batch"], weight_rule=config["weight_rule"], dynamics_rule=config["dynamics_rule"],
                         seed=config["seed"], train_nll=result["train_nll"], valid_nll=result["valid_nll"], test_nll=result["test_nll"],
                         alignment_rmse=float(np.sqrt(np.mean((history["train_nll"][1:] - baseline["train_nll"][1:]) ** 2))),
                         final_response_distance=float(np.linalg.norm(profiles[-1] - reference_profiles[-1]))))
    with (args.results / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for batch, w, d in sorted({(r["batch"], r["weight_rule"], r["dynamics_rule"]) for r in rows}):
        selected = [r for r in rows if (r["batch"], r["weight_rule"], r["dynamics_rule"]) == (batch, w, d)]
        item = dict(batch=batch, weight_rule=w, dynamics_rule=d, seeds=len(selected))
        for field in ("train_nll", "valid_nll", "test_nll", "alignment_rmse", "final_response_distance"):
            values = np.array([r[field] for r in selected])
            item[field] = float(values.mean())
            item[field + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregates.append(item)
        print(f"B={batch} ordinary={w} dynamics={d} valid={item['valid_nll']:.6f} +/- {item['valid_nll_std']:.6f}")
    (args.results / "aggregates.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
