#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.11")
from npfl138.datasets.modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout in the classifier.")
parser.add_argument("--epochs", default=60, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--modelnet", default=32, type=int, choices=[20, 32], help="ModelNet dimension.")
parser.add_argument("--patience", default=10, type=int, help="Early stopping patience.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="AdamW weight decay.")
parser.add_argument("--workers", default=0, type=int, help="DataLoader worker processes.")


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: ModelNet.Dataset, augment: bool = False) -> None:
        super().__init__(dataset)
        self._augment = augment

    def transform(self, example: ModelNet.Element):
        grid = example["grid"].to(torch.float32)
        if self._augment:
            # Random rotations around the vertical image axis preserve the class
            # and reduce the orientation bias of the small training set.
            rotations = torch.randint(0, 4, ()).item()
            if rotations:
                grid = torch.rot90(grid, rotations, dims=(-2, -1))
            if torch.rand(()) < 0.5:
                grid = torch.flip(grid, dims=(-1,))
        return grid, example["label"].to(torch.long)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._model = torch.nn.Sequential(
            torch.nn.Conv3d(ModelNet.C, 32, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256, pool=False),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(128, ModelNet.LABELS),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int, pool: bool = True) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = [
            torch.nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(output_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(output_channels),
            torch.nn.ReLU(),
        ]
        if pool:
            layers.append(torch.nn.MaxPool3d(2))
        return torch.nn.Sequential(*layers)

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        return self._model(grids)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    modelnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    train = Dataset(modelnet.train, augment=True).dataloader(
        args.batch_size, shuffle=True, seed=args.seed, num_workers=args.workers)
    dev = Dataset(modelnet.dev).dataloader(args.batch_size, num_workers=args.workers)
    test = Dataset(modelnet.test).dataloader(args.batch_size, num_workers=args.workers)

    model = Model(args)
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
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=ModelNet.LABELS)},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "3d_recognition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(dev, data_with_labels=True, console=0):
            print(prediction.argmax().item(), file=predictions_file)

    with open(os.path.join(logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(grid, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True, console=0):
            print(prediction.argmax().item(), file=predictions_file)

    if best_accuracy > float("-inf"):
        print(f"Best dev accuracy: {100 * best_accuracy:.2f}%")
        print(f"Latest dev prediction file: {os.path.join(logdir, '3d_recognition_dev.txt')}")
    print(f"Latest test prediction file: {os.path.join(logdir, '3d_recognition.txt')}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
