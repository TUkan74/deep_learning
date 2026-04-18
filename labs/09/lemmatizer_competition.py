#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer
from lemmatizer_attn import Model, TrainableDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--analyzer_postprocess", default=True, action=argparse.BooleanOptionalAction,
                    help="Snap predictions to closest analyzer lemma when possible.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=128, type=int, help="Character embedding dimension.")
parser.add_argument("--epochs", default=12, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--patience", default=3, type=int, help="Early stopping patience.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=0, type=int, help="Show examples during training.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--tie_embeddings", default=True, action=argparse.BooleanOptionalAction,
                    help="Tie decoder input and output embeddings.")


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, char_first in enumerate(first, start=1):
        current = [i]
        for j, char_second in enumerate(second, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_first != char_second),
            ))
        previous = current
    return previous[-1]


def choose_lemma(predicted: str, word: str, analyses: MorphoAnalyzer, use_analyzer: bool) -> str:
    if not use_analyzer:
        return predicted
    candidates = sorted({analysis.lemma for analysis in analyses.get(word)})
    if not candidates or predicted in candidates:
        return predicted
    return min(candidates, key=lambda candidate: (edit_distance(predicted, candidate), len(candidate), candidate))


def write_predictions(
    model: Model,
    dataset: MorphoDataset.Dataset,
    dataloader: torch.utils.data.DataLoader,
    output_path: str,
    analyses: MorphoAnalyzer,
    use_analyzer: bool,
) -> None:
    with open(output_path, "w", encoding="utf-8") as predictions_file:
        predictions = iter(model.predict(dataloader, data_with_labels=True))
        for sentence in dataset.words.strings:
            for word in sentence:
                lemma = "".join(dataset.lemmas.char_vocab.strings(next(predictions)))
                print(choose_lemma(lemma, word, analyses, use_analyzer), file=predictions_file)
            print(file=predictions_file)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}", **vars(args))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    train = TrainableDataset(morpho.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, training=False).dataloader(batch_size=args.batch_size)

    model = Model(args, morpho.train)
    best_accuracy = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    def keep_best(model: Model, epoch: int, logs: dict[str, float]) -> None | npfl138.StopTraining:
        nonlocal best_accuracy, best_state_dict, epochs_without_improvement
        if logs["dev:accuracy"] > best_accuracy:
            best_accuracy = logs["dev:accuracy"]
            best_state_dict = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                return npfl138.STOP_TRAINING
        return None

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    write_predictions(
        model, morpho.dev, dev, os.path.join(logdir, "lemmatizer_competition_dev.txt"),
        analyses, args.analyzer_postprocess,
    )
    write_predictions(
        model, morpho.test, test, os.path.join(logdir, "lemmatizer_competition.txt"),
        analyses, args.analyzer_postprocess,
    )

    if best_accuracy > float("-inf"):
        print(f"Best dev accuracy before analyzer postprocessing: {100 * best_accuracy:.2f}%")
    print(f"Latest dev prediction file: {os.path.join(logdir, 'lemmatizer_competition_dev.txt')}")
    print(f"Latest test prediction file: {os.path.join(logdir, 'lemmatizer_competition.txt')}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
