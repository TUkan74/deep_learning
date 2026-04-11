#!/usr/bin/env python3
import argparse
from pathlib import Path

import npfl138
npfl138.require_version("2526.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs


def main(args: argparse.Namespace) -> None:
    root = Path(args.logdir)
    if not root.exists():
        raise FileNotFoundError(f"Log directory '{root}' does not exist.")

    common_voice = CommonVoiceCs()
    candidates: list[tuple[float, Path, Path]] = []

    for dev_path in root.rglob("speech_recognition_dev.txt"):
        test_path = dev_path.with_name("speech_recognition.txt")
        if not test_path.exists():
            continue
        with dev_path.open("r", encoding="utf-8-sig") as predictions_file:
            edit_distance = CommonVoiceCs.evaluate_file(common_voice.dev, predictions_file)
        candidates.append((edit_distance, dev_path, test_path))

    if not candidates:
        print("Speech recognition: no completed runs found.")
        return

    candidates.sort(key=lambda item: item[0])
    print("Submit this .py file:")
    print("- labs/08/speech_recognition.py")
    print()
    print("Speech recognition top runs by dev edit distance:")
    for i, (score, dev_path, test_path) in enumerate(candidates[:args.top], start=1):
        print(f"{i}. {100 * score:.2f}%")
        print(f"   dev:  {dev_path}")
        print(f"   test: {test_path}")

    best_score, _, best_test = candidates[0]
    print(f"Speech recognition submit txt: {best_test}")
    print(f"Speech recognition best dev edit distance: {100 * best_score:.2f}%")
    if best_score > 0.45:
        print("Warning: above the 45% target.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="Directory containing training runs.")
    parser.add_argument("--top", default=5, type=int, help="How many runs to print.")
    main(parser.parse_args())
