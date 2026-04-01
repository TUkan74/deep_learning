#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS


def rank_classification(root: Path, top: int) -> None:
    cags = CAGS(decode_on_demand=True)
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("cags_classification_dev.txt"):
        test_path = dev_path.with_name("cags_classification.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = CAGS.evaluate_classification_file(cags.dev, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    if not candidates:
        print("Classification: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Classification top runs by dev accuracy:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Classification submit txt: {best_test}")
    print(f"Classification best dev: {100 * best_score:.2f}%")
    if best_score < 0.93:
        print("Warning: below 93% dev target.")
    print()


def rank_segmentation(root: Path, top: int) -> None:
    cags = CAGS(decode_on_demand=True)
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("cags_segmentation_dev.txt"):
        test_path = dev_path.with_name("cags_segmentation.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            iou = CAGS.evaluate_segmentation_file(cags.dev, predictions_file)
        candidates.append((iou, dev_path, test_path))

    if not candidates:
        print("Segmentation: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Segmentation top runs by dev IoU:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Segmentation submit txt: {best_test}")
    print(f"Segmentation best dev IoU: {100 * best_score:.2f}%")
    if best_score < 0.87:
        print("Warning: below 87% dev IoU target.")
    print()


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    print("Submit these .py files:")
    print("- labs/05/cags_classification.py")
    print("- labs/05/cags_segmentation.py")
    print()

    rank_classification(root, args.top)
    rank_segmentation(root, args.top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
