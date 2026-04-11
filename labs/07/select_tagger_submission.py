#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.7")
from npfl138.datasets.morpho_dataset import MorphoDataset


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    morpho = MorphoDataset("czech_pdt")
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("tagger_competition_dev.txt"):
        test_path = dev_path.with_name("tagger_competition.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = MorphoDataset.evaluate_file(morpho.dev.tags, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    if not candidates:
        print("Tagger competition: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Submit this .py file:")
    print("- labs/07/tagger_competition.py")
    print()
    print("Tagger competition top runs by dev accuracy:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:args.top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Tagger competition submit txt: {best_test}")
    print(f"Tagger competition best dev accuracy: {100 * best_score:.2f}%")
    if best_score < 0.93:
        print("Warning: below 93.0% dev target.")
    if best_score < 0.9635:
        print("Note: below the 96.35% bonus threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
