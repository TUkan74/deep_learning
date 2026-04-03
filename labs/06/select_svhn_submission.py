#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.6")
from npfl138.datasets.svhn import SVHN


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    svhn = SVHN(decode_on_demand=True)
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("svhn_competition_dev.txt"):
        test_path = dev_path.with_name("svhn_competition.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = SVHN.evaluate_file(svhn.dev, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    if not candidates:
        print("SVHN: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("SVHN top runs by dev accuracy:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:args.top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"SVHN submit txt: {best_test}")
    print(f"SVHN best dev accuracy: {100 * best_score:.2f}%")
    if best_score < 0.35:
        print("Warning: below the rough 35% dev target from the assignment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
