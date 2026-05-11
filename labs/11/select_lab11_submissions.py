#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import npfl138
npfl138.require_version("2526.11")
from npfl138.datasets.modelnet import ModelNet


def infer_modelnet_dim(path: Path, fallback: int) -> int:
    match = re.search(r"(?:^|[,/-])m=(20|32)(?:,|$)", str(path))
    return int(match.group(1)) if match else fallback


def rank_3d_recognition(root: Path, top: int, fallback_dim: int) -> None:
    datasets: dict[int, ModelNet] = {}
    candidates: list[tuple[float, int, Path, Path]] = []

    for dev_path in root.rglob("3d_recognition_dev.txt"):
        test_path = dev_path.with_name("3d_recognition.txt")
        if not test_path.exists():
            continue
        dim = infer_modelnet_dim(dev_path, fallback_dim)
        datasets.setdefault(dim, ModelNet(dim))
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = ModelNet.evaluate_file(datasets[dim].dev, predictions_file)
        candidates.append((accuracy, dim, dev_path, test_path))

    if not candidates:
        print("3D recognition: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("3D recognition top runs by dev accuracy:")
    for i, (score, dim, dev_path, test_path) in enumerate(candidates[:top], start=1):
        print(f"{i}. {100 * score:.2f}% (ModelNet{dim})")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, best_dim, _, best_test = candidates[0]
    print(f"3D recognition submit txt: {best_test}")
    print(f"3D recognition best dev accuracy: {100 * best_score:.2f}% (ModelNet{best_dim})")
    if best_score < 0.89:
        print("Warning: below the 89% target from the assignment.")
    print()


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    print("Submit these .py files:")
    print("- labs/11/reinforce.py")
    print("- labs/11/reinforce_baseline.py")
    print("- labs/11/reinforce_pixels.py")
    print("- labs/11/3d_recognition.py")
    print()

    rank_3d_recognition(root, args.top, args.modelnet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--modelnet", default=32, type=int, choices=[20, 32],
                        help="ModelNet dimension to use if it cannot be inferred from a log path.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
