#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn.functional as F

import npfl138
npfl138.require_version("2526.12")
from npfl138.datasets.homr_dataset import HOMRDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
PAD = 0

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--cnn_dim", default=192, type=int, help="CNN output channels.")
parser.add_argument("--decode_on_demand", default=True, action=argparse.BooleanOptionalAction,
                    help="Decode PNG images lazily instead of keeping all decoded images in memory.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--height", default=128, type=int, help="Image height after resizing.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--max_width", default=3000, type=int, help="Maximum resized image width.")
parser.add_argument("--patience", default=5, type=int, help="Early stopping patience.")
parser.add_argument("--rnn_dim", default=256, type=int, help="Recurrent hidden size.")
parser.add_argument("--rnn_layers", default=3, type=int, help="Number of recurrent layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--workers", default=0, type=int, help="DataLoader worker processes.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._cnn = torch.nn.Sequential(
            self._conv_block(HOMRDataset.C, 32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(32, 64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(64, 128),
            torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            self._conv_block(128, args.cnn_dim),
            torch.nn.Dropout2d(args.dropout),
        )
        self._rnn = torch.nn.GRU(
            args.cnn_dim,
            args.rnn_dim,
            num_layers=args.rnn_layers,
            dropout=args.dropout if args.rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self._output_layer = torch.nn.Linear(2 * args.rnn_dim, HOMRDataset.MARKS)
        self._ctc_loss = torch.nn.CTCLoss(blank=PAD, zero_infinity=True)

    @staticmethod
    def _conv_block(input_channels: int, output_channels: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
        )

    def forward(self, images: torch.Tensor, image_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._cnn(images)
        hidden = hidden.mean(dim=2).permute(0, 2, 1)
        output_lengths = torch.clamp(image_lengths // 4, min=1, max=hidden.shape[1])

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden, output_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed, _ = self._rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        return self._output_layer(hidden).permute(0, 2, 1), output_lengths

    def compute_loss(
        self, y_pred: tuple[torch.Tensor, torch.Tensor], y_true: torch.Tensor,
        images: torch.Tensor, image_lengths: torch.Tensor,
    ) -> torch.Tensor:
        logits, output_lengths = y_pred
        target_lengths = (y_true != PAD).sum(dim=1)
        return self._ctc_loss(
            logits.log_softmax(dim=1).permute(2, 0, 1), y_true, output_lengths.cpu(), target_lengths.cpu(),
        )

    def ctc_decoding(self, logits: torch.Tensor, output_lengths: torch.Tensor) -> list[list[int]]:
        best_labels = logits.argmax(dim=1)
        predictions: list[list[int]] = []
        for labels, length in zip(best_labels, output_lengths):
            labels = labels[:int(length)]
            if len(labels) == 0:
                predictions.append([])
                continue
            keep = torch.ones_like(labels, dtype=torch.bool)
            keep[1:] = labels[1:] != labels[:-1]
            labels = labels[keep]
            labels = labels[labels != PAD]
            predictions.append(labels.tolist())
        return predictions

    def compute_metrics(
        self, y_pred: tuple[torch.Tensor, torch.Tensor], y_true: torch.Tensor,
        images: torch.Tensor, image_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.training:
            return {}
        predictions = self.ctc_decoding(*y_pred)
        gold = [marks[marks != PAD].tolist() for marks in y_true]
        self.metrics["edit_distance"].update(predictions, gold)
        return {"edit_distance": self.metrics["edit_distance"]}

    def predict_step(self, xs):
        with torch.no_grad():
            yield from self.ctc_decoding(*self.forward(*xs))


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: HOMRDataset.Dataset, args: argparse.Namespace) -> None:
        super().__init__(dataset)
        self._height = args.height
        self._max_width = args.max_width

    def transform(self, example: HOMRDataset.Element):
        image = 1 - example["image"].to(torch.float32) / 255
        return image, example["marks"].to(torch.long)

    def collate(self, batch):
        images, marks = zip(*batch)
        resized_images, image_lengths = [], []
        for image in images:
            height, width = image.shape[-2:]
            resized_width = max(1, round(width * self._height / height))
            if self._max_width:
                resized_width = min(resized_width, self._max_width)
            image = F.interpolate(
                image.unsqueeze(0), size=(self._height, resized_width), mode="bilinear", align_corners=False,
            ).squeeze(0)
            resized_images.append(image)
            image_lengths.append(resized_width)

        max_width = max(image_lengths)
        padded_images = torch.zeros(len(resized_images), HOMRDataset.C, self._height, max_width)
        for i, image in enumerate(resized_images):
            padded_images[i, :, :, :image.shape[-1]] = image

        padded_marks = torch.nn.utils.rnn.pad_sequence(marks, batch_first=True, padding_value=PAD)
        return (padded_images, torch.tensor(image_lengths, dtype=torch.int64)), padded_marks


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[1, HEIGHT, WIDTH]` tensor of `torch.uint8` values in [0-255] range,
    # - "marks", a `[num_marks]` tensor with indices of marks on the image.
    # Using `decode_on_demand=True` loads just the raw dataset (~500MB of undecoded PNG images)
    # and then decodes them on every access. Using `decode_on_demand=False` decodes the images
    # during loading, resulting in much faster access, but requires ~5GB of memory.
    homr = HOMRDataset(decode_on_demand=args.decode_on_demand)

    # TODO: Create the model and train it.
    train = TrainableDataset(homr.train, args).dataloader(
        args.batch_size, shuffle=True, seed=args.seed, num_workers=args.workers)
    dev = TrainableDataset(homr.dev, args).dataloader(args.batch_size, num_workers=args.workers)
    test = TrainableDataset(homr.test, args).dataloader(args.batch_size, num_workers=args.workers)

    model = Model(args)
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
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate),
        metrics={"edit_distance": HOMRDataset.EditDistanceMetric(ignore_index=PAD)},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "homr_competition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        for sequence in model.predict(dev, data_with_labels=True, console=0):
            print(" ".join(HOMRDataset.MARKS_VOCAB.strings(sequence)), file=predictions_file)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    with open(os.path.join(logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions = model.predict(test, data_with_labels=True, console=0)

        for sequence in predictions:
            print(" ".join(HOMRDataset.MARKS_VOCAB.strings(sequence)), file=predictions_file)

    if best_edit_distance < float("inf"):
        print(f"Best dev edit distance: {100 * best_edit_distance:.3f}%")
        print(f"Latest dev prediction file: {os.path.join(logdir, 'homr_competition_dev.txt')}")
    print(f"Latest test prediction file: {os.path.join(logdir, 'homr_competition.txt')}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
