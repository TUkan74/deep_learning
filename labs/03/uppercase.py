#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.3")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=0, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
parser.add_argument("--embedding_dim", default=64, type=int, help="Character embedding dimension.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[512, 256], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--patience", default=4, type=int, help="Early stopping patience.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")


class Dataset(torch.utils.data.Dataset):
    # A dataset must always implement at least `__len__` and `__getitem__`.
    def __init__(self, uppercase_dataset: UppercaseData.Dataset):
        self.windows = uppercase_dataset.windows
        self.labels = uppercase_dataset.labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index], self.labels[index]

    # When a dataset implements `__getitems__`, this method is used to generate whole batches in a single call.
    # However, `__getitems__` is expected to return a list of items that are later collated together.
    # For maximum speedup, we already return a whole batch from `__getitems__` and implement a trivial `collate`.
    def __getitems__(self, indices):
        indices = torch.as_tensor(indices)
        return self.windows[indices], self.labels[indices]

    @staticmethod
    def collate(batch):
        return batch


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, alphabet_size: int):
        super().__init__()
        self._args = args

        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a Dense layer)
        #   and flattening afterwards, or suitably using `torch.nn.EmbeddingBag`.
        self._embedding = torch.nn.Embedding(alphabet_size, args.embedding_dim, padding_idx=0)
        self._input_dropout = torch.nn.Dropout(args.dropout)

        layers: list[torch.nn.Module] = []
        features = (2 * args.window + 1) * args.embedding_dim
        for hidden_layer in args.hidden_layers:
            layers.append(torch.nn.Linear(features, hidden_layer))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(args.dropout))
            features = hidden_layer
        layers.append(torch.nn.Linear(features, UppercaseData.LABELS))
        self._classifier = torch.nn.Sequential(*layers)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        windows = self._embedding(windows)
        windows = windows.reshape(windows.shape[0], -1)
        windows = self._input_dropout(windows)
        return self._classifier(windows)


def find_best_threshold(probabilities: np.ndarray, gold_labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(probabilities)[::-1]
    probabilities = probabilities[order]
    gold_labels = gold_labels[order].astype(np.int64)

    correct = int(np.sum(gold_labels == 0))
    best_correct = correct
    best_threshold = float(np.nextafter(probabilities[0], np.inf)) if len(probabilities) else 0.5

    index = 0
    while index < len(probabilities):
        probability = probabilities[index]
        while index < len(probabilities) and probabilities[index] == probability:
            correct += 1 if gold_labels[index] == 1 else -1
            index += 1

        next_probability = probabilities[index] if index < len(probabilities) else 0.0
        threshold = float((probability + next_probability) / 2)
        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    return best_threshold, best_correct / len(gold_labels)


def uppercase_text(text: str, uppercase_predictions: np.ndarray) -> str:
    return "".join(character.upper() if uppercase else character for character, uppercase in zip(text, uppercase_predictions))


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data and create windows of integral character indices and integral labels.
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    train = torch.utils.data.DataLoader(
        Dataset(uppercase_data.train), args.batch_size, collate_fn=Dataset.collate, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(uppercase_data.dev), args.batch_size, collate_fn=Dataset.collate)
    test = torch.utils.data.DataLoader(Dataset(uppercase_data.test), args.batch_size, collate_fn=Dataset.collate)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    model = Model(args, len(uppercase_data.train.alphabet))
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=UppercaseData.LABELS)},
        logdir=logdir,
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_accuracy = float("-inf")
    epochs_without_improvement = 0

    def callback(model: Model, epoch: int, logs: dict[str, float]):
        nonlocal best_state_dict, best_accuracy, epochs_without_improvement

        if logs["dev:accuracy"] > best_accuracy:
            best_accuracy = logs["dev:accuracy"]
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            return npfl138.STOP_TRAINING

    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[callback])
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    dev_logits = model.predict_tensor(dev, data_with_labels=True, console=0)
    dev_probabilities = torch.softmax(dev_logits, dim=-1)[:, 1].cpu().numpy()
    threshold, threshold_accuracy = find_best_threshold(dev_probabilities, uppercase_data.dev.labels.cpu().numpy())
    print(f"Best threshold {threshold:.4f}, dev accuracy {100 * threshold_accuracy:.2f}%", flush=True)

    # TODO: Generate correctly capitalized test set and write the result to `predictions_file`,
    # which is by default `uppercase_test.txt` in the `logdir` directory).
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        # We start by generating the network test set predictions; if you modified the `test` dataloader
        # or your model does not process the dataset windows, you might need to adjust the following line.
        predictions = model.predict_tensor(test, data_with_labels=True, console=0)

        # Now you need to utilize the network predictions and the unannotated test data (lowercased text)
        # available in `uppercase_data.test.text` to produce capitalized text and print it to the `predictions_file`.
        predictions = torch.softmax(predictions, dim=-1)[:, 1].cpu().numpy() >= threshold
        print(uppercase_text(uppercase_data.test.text, predictions), file=predictions_file, end="")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
