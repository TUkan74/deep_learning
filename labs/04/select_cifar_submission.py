#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.cifar10 import CIFAR10


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    cifar = CIFAR10()
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("cifar_competition_dev.txt"):
        test_path = dev_path.with_name("cifar_competition_test.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = CIFAR10.evaluate_file(cifar.dev, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    if not candidates:
        print("No completed CIFAR competition runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)

    print("Top runs by dev accuracy:")
    for index, (accuracy, dev_path, test_path) in enumerate(candidates[:args.top], start=1):
        print(f"{index}. {100 * accuracy:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_accuracy, best_dev, best_test = candidates[0]
    print()
    print("Submit these files:")
    print(f"- txt: {best_test}")
    print(f"- py:  {Path('labs/04/cifar_competition.py')}")
    print(f"Best dev accuracy: {100 * best_accuracy:.2f}%")
    if best_accuracy < 0.85:
        print("Warning: this is below the rough ~85% dev target mentioned in the assignment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
