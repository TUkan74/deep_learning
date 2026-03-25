#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics
from torchvision.transforms import v2

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--channels", default=64, type=int, help="Width of the first convolution block.")
parser.add_argument("--classifier_dropout", default=0.3, type=float, help="Dropout before the classifier.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout inside residual blocks.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="AdamW weight decay.")


class Dataset(npfl138.TransformedDataset):
    _NORMALIZE = v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))

    def __init__(self, dataset: CIFAR10.Dataset, *, training: bool = False) -> None:
        super().__init__(dataset)
        self._training = training
        self._augmentation = v2.Compose([
            v2.RandomCrop((CIFAR10.H, CIFAR10.W), padding=4),
            v2.RandomHorizontalFlip(),
        ]) if training else None

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = example["image"].to(torch.float32) / 255
        if self._augmentation is not None:
            image = self._augmentation(image)
        image = self._NORMALIZE(image)
        return image, example["label"]


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self._block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(dropout) if dropout else torch.nn.Identity(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        self._shortcut = torch.nn.Identity() if stride == 1 and in_channels == out_channels else torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        self._activation = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._activation(self._block(inputs) + self._shortcut(inputs))


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        c1, c2, c3 = args.channels, args.channels * 2, args.channels * 4
        super().__init__(torch.nn.Sequential(
            torch.nn.Conv2d(CIFAR10.C, c1, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(c1),
            torch.nn.ReLU(),
            ResidualBlock(c1, c1, dropout=args.dropout),
            ResidualBlock(c1, c2, stride=2, dropout=args.dropout),
            ResidualBlock(c2, c2, dropout=args.dropout),
            ResidualBlock(c2, c3, stride=2, dropout=args.dropout),
            ResidualBlock(c3, c3, dropout=args.dropout),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(args.classifier_dropout),
            torch.nn.Linear(c3, CIFAR10.LABELS),
        ))


def write_predictions(
    model: Model, dataloader: torch.utils.data.DataLoader, path: str, *, data_with_labels: bool = True,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(dataloader, data_with_labels=data_with_labels, console=0):
            print(prediction.argmax().item(), file=predictions_file)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    cifar = CIFAR10()

    # TODO: Create the model and train it.
    train = Dataset(cifar.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = Dataset(cifar.dev).dataloader(batch_size=args.batch_size)
    test = Dataset(cifar.test).dataloader(batch_size=args.batch_size)

    model = Model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train))
    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CIFAR10.LABELS)},
        logdir=logdir,
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_accuracy = float("-inf")

    def callback(model: Model, epoch: int, logs: dict[str, float]):
        nonlocal best_state_dict, best_accuracy

        if logs["dev:accuracy"] > best_accuracy:
            best_accuracy = logs["dev:accuracy"]
            best_state_dict = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}

    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[callback])
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Using best dev accuracy: {100 * best_accuracy:.2f}%", flush=True)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    write_predictions(model, dev, os.path.join(logdir, "cifar_competition_dev.txt"))
    write_predictions(model, test, os.path.join(logdir, "cifar_competition_test.txt"))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
