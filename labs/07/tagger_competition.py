#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.7")
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--patience", default=2, type=int, help="Early stopping patience.")
parser.add_argument("--rnn", default="GRU", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=96, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")


class Model(npfl138.TrainableModule):
    class MaskElements(torch.nn.Module):
        def __init__(self, mask_probability: float, mask_value: int):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask = torch.rand_like(inputs, dtype=torch.float32) < self._mask_probability
                mask &= inputs != MorphoDataset.PAD
                inputs = torch.where(mask, torch.full_like(inputs, self._mask_value), inputs)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._rnn_dim = args.rnn_dim

        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK)
        self._char_embedding = torch.nn.Embedding(len(train.words.char_vocab), args.cle_dim)
        self._char_rnn = torch.nn.GRU(args.cle_dim, args.cle_dim, batch_first=True, bidirectional=True)
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)
        self._word_rnn = getattr(torch.nn, args.rnn)(
            args.we_dim + 2 * args.cle_dim, args.rnn_dim, batch_first=True, bidirectional=True,
        )
        self._output_layer = torch.nn.Linear(args.rnn_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor) -> torch.Tensor:
        word_lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        char_lengths = (unique_words != MorphoDataset.PAD).sum(dim=1).cpu()

        hidden = self._word_embedding(self._word_masking(word_ids))

        cle = self._char_embedding(unique_words)
        packed_cle = torch.nn.utils.rnn.pack_padded_sequence(
            cle, char_lengths, batch_first=True, enforce_sorted=False,
        )
        _, cle = self._char_rnn(packed_cle)
        cle = torch.cat([cle[-2], cle[-1]], dim=-1)
        cle = torch.nn.functional.embedding(word_indices, cle)

        hidden = torch.cat([hidden, cle], dim=-1)
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
            hidden, word_lengths, batch_first=True, enforce_sorted=False,
        )
        packed_words, _ = self._word_rnn(packed_words)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_words, batch_first=True)
        hidden = hidden[:, :, :self._rnn_dim] + hidden[:, :, self._rnn_dim:]
        return self._output_layer(hidden).permute(0, 2, 1)


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.int64)
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"]), dtype=torch.int64)
        return word_ids, example["words"], tag_ids

    def collate(self, batch):
        word_ids, words, tag_ids = zip(*batch)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        unique_words, words_indices = self.dataset.cle_batch(list(words))
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        return (word_ids, unique_words, words_indices), tag_ids


def choose_tag(
    scores: torch.Tensor,
    word: str,
    train: MorphoDataset.Dataset,
    analyses: MorphoAnalyzer,
) -> str:
    candidate_tags = []
    for analysis in analyses.get(word):
        index = train.tags.string_vocab.index(analysis.tag)
        if index is None:
            continue
        if train.tags.string_vocab.string(index) == analysis.tag:
            candidate_tags.append(index)

    if candidate_tags:
        best_index = max(candidate_tags, key=lambda index: float(scores[index]))
    else:
        best_index = int(scores.argmax())
    return train.tags.string_vocab.string(best_index)


def write_predictions(
    model: Model,
    dataset: MorphoDataset.Dataset,
    dataloader: torch.utils.data.DataLoader,
    output_path: str,
    analyses: MorphoAnalyzer,
    train: MorphoDataset.Dataset,
) -> None:
    with open(output_path, "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(dataloader, data_with_labels=True, as_numpy=False)

        for predicted_tags, words in zip(predictions, dataset.words.strings):
            predicted_tags = predicted_tags[:, :len(words)]
            for i, word in enumerate(words):
                print(choose_tag(predicted_tags[:, i], word, train, analyses), file=predictions_file)
            print(file=predictions_file)


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}", **vars(args))

    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test).dataloader(batch_size=args.batch_size)

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
        metrics={"accuracy": torchmetrics.Accuracy(
            "multiclass", num_classes=len(morpho.train.tags.string_vocab), ignore_index=morpho.PAD,
        )},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    os.makedirs(logdir, exist_ok=True)
    write_predictions(
        model, morpho.dev, dev, os.path.join(logdir, "tagger_competition_dev.txt"), analyses, morpho.train,
    )
    write_predictions(
        model, morpho.test, test, os.path.join(logdir, "tagger_competition.txt"), analyses, morpho.train,
    )

    if best_accuracy > float("-inf"):
        print(f"Best dev accuracy: {100 * best_accuracy:.2f}%")
        print(f"Latest dev prediction file: {os.path.join(logdir, 'tagger_competition_dev.txt')}")
    print(f"Latest test prediction file: {os.path.join(logdir, 'tagger_competition.txt')}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
