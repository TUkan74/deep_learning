#!/usr/bin/env python3
import argparse
import os

import torch
import torchaudio.models.decoder

import npfl138
npfl138.require_version("2526.8.1")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout between recurrent layers.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--patience", default=3, type=int, help="Early stopping patience.")
parser.add_argument("--rnn_dim", default=256, type=int, help="Recurrent hidden state size.")
parser.add_argument("--rnn_layers", default=3, type=int, help="Number of recurrent layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        self._rnn = torch.nn.GRU(
            CommonVoiceCs.MFCC_DIM,
            args.rnn_dim,
            num_layers=args.rnn_layers,
            dropout=args.dropout if args.rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self._output_layer = torch.nn.Linear(2 * args.rnn_dim, CommonVoiceCs.LETTERS)
        self._ctc_loss = torch.nn.CTCLoss(blank=CommonVoiceCs.PAD, zero_infinity=True)

    def forward(self, mfccs: torch.Tensor, mfcc_lengths: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the output of the model.
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            mfccs, mfcc_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed, _ = self._rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        return self._output_layer(hidden).permute(0, 2, 1)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs: torch.Tensor, mfcc_lengths: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        target_lengths = (y_true != CommonVoiceCs.PAD).sum(dim=1)
        return self._ctc_loss(
            y_pred.log_softmax(dim=1).permute(2, 0, 1), y_true, mfcc_lengths, target_lengths,
        )

    def ctc_decoding(self, y_pred: torch.Tensor, mfccs: torch.Tensor, mfcc_lengths: torch.Tensor) -> list[list[int]]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        best_labels = y_pred.argmax(dim=1)
        predictions: list[list[int]] = []
        for labels, length in zip(best_labels, mfcc_lengths):
            labels = labels[:length]
            if len(labels) == 0:
                predictions.append([])
                continue
            keep = torch.ones_like(labels, dtype=torch.bool)
            keep[1:] = labels[1:] != labels[:-1]
            labels = labels[keep]
            labels = labels[labels != CommonVoiceCs.PAD]
            predictions.append(labels.tolist())
        return predictions

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs: torch.Tensor, mfcc_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        if self.training:
            return {}

        predictions = self.ctc_decoding(y_pred, mfccs, mfcc_lengths)
        gold = [sentence[sentence != CommonVoiceCs.PAD].tolist() for sentence in y_true]
        self.metrics["edit_distance"].update(predictions, gold)
        return {"edit_distance": self.metrics["edit_distance"]}

    def predict_step(self, xs):
        with torch.no_grad():
            # Perform constrained decoding.
            yield from self.ctc_decoding(self.forward(*xs), *xs)


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # You can use `CommonVoiceCs.LETTER_NAMES : list[str]` or `CommonVoiceCs.LETTERS_VOCAB : npfl138.Vocabulary`
        # to convert between letters and their indices. While the letters do not explicitly contain
        # a blank token, the [PAD] token can be employed as one.
        return (
            example["mfccs"].to(torch.float32),
            torch.tensor(CommonVoiceCs.LETTERS_VOCAB.indices(example["sentence"]), dtype=torch.int64),
        )

    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        mfccs, sentences = zip(*batch)
        mfcc_lengths = torch.tensor([len(mfcc) for mfcc in mfccs], dtype=torch.int64)
        mfccs = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
        sentences = torch.nn.utils.rnn.pad_sequence(
            sentences, batch_first=True, padding_value=CommonVoiceCs.PAD,
        )
        return (mfccs, mfcc_lengths), sentences


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}", **vars(args))

    # Load the data.
    common_voice = CommonVoiceCs()

    train = Dataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = Dataset(common_voice.dev).dataloader(args.batch_size)
    test = Dataset(common_voice.test).dataloader(args.batch_size)

    # TODO: Create the model and train it. The `Model.compute_metrics` method assumes you
    # passed the following metric to the `configure` method under the name "edit_distance":
    #   CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)
    model = Model(args, common_voice.train)
    best_edit_distance = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    def keep_best(model: Model, epoch: int, logs: dict[str, float]) -> None | npfl138.StopTraining:
        nonlocal best_edit_distance, best_state_dict, epochs_without_improvement
        if logs["dev:edit_distance"] < best_edit_distance:
            best_edit_distance = logs["dev:edit_distance"]
            best_state_dict = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                return npfl138.STOP_TRAINING
        return None

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        metrics={"edit_distance": CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "speech_recognition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        for sentence in model.predict(dev, data_with_labels=True):
            print("".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence)), file=predictions_file)

    # Generate test set annotations, but in `model.logdir` to allow parallel execution.
    with open(os.path.join(logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test, data_with_labels=True)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence)), file=predictions_file)

    if best_edit_distance < float("inf"):
        print(f"Best dev edit distance: {100 * best_edit_distance:.2f}%")
    print(f"Latest dev prediction file: {os.path.join(logdir, 'speech_recognition_dev.txt')}")
    print(f"Latest test prediction file: {os.path.join(logdir, 'speech_recognition.txt')}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
