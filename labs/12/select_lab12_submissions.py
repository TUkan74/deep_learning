#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.12")
from npfl138.datasets.homr_dataset import HOMRDataset


def rank_homr(root: Path, top: int) -> None:
    homr = HOMRDataset(decode_on_demand=True)
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("homr_competition_dev.txt"):
        test_path = dev_path.with_name("homr_competition.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            edit_distance = HOMRDataset.evaluate_file(homr.dev, predictions_file)
        candidates.append((edit_distance, dev_path, test_path))

    if not candidates:
        print("HOMR competition: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0])
    print("HOMR competition top runs by dev edit distance:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:top], start=1):
        print(f"{i}. {100 * score:.3f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"HOMR competition submit txt: {best_test}")
    print(f"HOMR competition best dev edit distance: {100 * best_score:.3f}%")
    if best_score > 0.03:
        print("Warning: above the 3% target from the assignment.")
    print()


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    print("Submit these .py files:")
    print("- labs/12/vae.py")
    print("- labs/12/gan.py")
    print("- labs/12/dcgan.py")
    print("- labs/12/homr_competition.py")
    print()

    rank_homr(root, args.top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
