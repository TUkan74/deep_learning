#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset
from npfl138.datasets.text_classification_dataset import TextClassificationDataset


def rank_reading_comprehension(root: Path, top: int) -> None:
    dataset = ReadingComprehensionDataset()
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("reading_comprehension_dev.txt"):
        test_path = dev_path.with_name("reading_comprehension.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = ReadingComprehensionDataset.evaluate_file(dataset.dev, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    if not candidates:
        print("Reading comprehension: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Reading comprehension top runs by dev accuracy:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Reading comprehension submit txt: {best_test}")
    print(f"Reading comprehension best dev accuracy: {100 * best_score:.2f}%")
    if best_score < 0.62:
        print("Warning: below the rough 62% dev target from the assignment.")
    print()


def rank_sentiment_analysis(root: Path, top: int) -> None:
    dataset = TextClassificationDataset("czech_facebook")
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("sentiment_analysis_dev.txt"):
        test_path = dev_path.with_name("sentiment_analysis.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            accuracy = TextClassificationDataset.evaluate_file(dataset.dev, predictions_file)
        candidates.append((accuracy, dev_path, test_path))

    if not candidates:
        print("Sentiment analysis: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Sentiment analysis top runs by dev accuracy:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Sentiment analysis submit txt: {best_test}")
    print(f"Sentiment analysis best dev accuracy: {100 * best_score:.2f}%")
    if best_score < 0.77:
        print("Warning: below the 77% target from the assignment.")
    print()


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    print("Submit these .py files:")
    print("- labs/10/reading_comprehension.py")
    print("- labs/10/sentiment_analysis.py")
    print("- labs/10/tagger_transformer.py")
    print()

    rank_reading_comprehension(root, args.top)
    rank_sentiment_analysis(root, args.top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
