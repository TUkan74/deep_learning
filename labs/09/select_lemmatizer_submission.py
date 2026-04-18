#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.9")
from npfl138.datasets.morpho_dataset import MorphoDataset


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    morpho = MorphoDataset("czech_pdt")
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("lemmatizer_competition_dev.txt"):
        test_path = dev_path.with_name("lemmatizer_competition.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = MorphoDataset.evaluate_file(morpho.dev.lemmas, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    print("Submit these .py files:")
    print("- labs/09/lemmatizer_competition.py")
    print("- labs/09/lemmatizer_attn.py")
    print()

    if not candidates:
        print("Lemmatizer competition: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Lemmatizer top runs by dev lemma accuracy:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:args.top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Lemmatizer submit txt: {best_test}")
    print(f"Lemmatizer best dev accuracy: {100 * best_score:.2f}%")
    if best_score < 0.97:
        print("Warning: below the 97% competition target.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
