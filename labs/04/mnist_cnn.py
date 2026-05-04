#!/usr/bin/env python3
import argparse

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


class ResidualBlock(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self._block = block

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self._block(inputs)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-channels-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of channels, kernel size, stride and padding.
        # - `CB-channels-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default padding of 0 (the "valid" padding).
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a fully connected layer with ReLU activation and the
        #   specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # To implement the residual connections, you can use various approaches, for example:
        # - you can create a specialized `torch.nn.Module` subclass representing a residual
        #   connection that gets the inside layers as an argument, and implement its forward call.
        #   This allows you to have the whole network in a single `torch.nn.Sequential`.
        # - you could represent the model module as a `torch.nn.ModuleList` of `torch.nn.Sequential`s,
        #   each representing one user-specified layer, keep track of the positions of residual
        #   connections, and manually perform them in the forward pass.
        #
        # It might be difficult to compute the number of features after the `F` layer. You can
        # nevertheless use the `torch.nn.LazyLinear`, `torch.nn.LazyConv2d`, and `torch.nn.LazyBatchNorm2d`
        # layers, which do not require the number of input features to be specified in the constructor.
        # During `__init__`, these layers do not allocate their parameters, and only do so when
        # they are first called on a tensor, at which point the number of input features is known.
        # During this first call they also change themselves to the corresponding `torch.nn.Linear` etc.
        architecture = args.cnn or "F"
        layers: list[torch.nn.Module] = []
        for specification in self._split_architecture(architecture):
            layers.extend(self._create_layers(specification))

        # TODO: Finally, add the final Linear output layer with `MNIST.LABELS` units.
        layers.append(torch.nn.LazyLinear(MNIST.LABELS))
        super().__init__(torch.nn.Sequential(*layers))

    @staticmethod
    def _split_architecture(specification: str) -> list[str]:
        layers, start, depth = [], 0, 0
        for i, char in enumerate(specification):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            elif char == "," and depth == 0:
                layers.append(specification[start:i])
                start = i + 1
        layers.append(specification[start:])
        return [layer.strip() for layer in layers if layer.strip()]

    @staticmethod
    def _parse_padding(padding: str) -> str | int:
        if padding == "same":
            return "same"
        if padding == "valid":
            return 0
        return int(padding)

    @classmethod
    def _create_layers(cls, specification: str) -> list[torch.nn.Module]:
        if specification.startswith("R-[") and specification.endswith("]"):
            residual_layers = []
            for inner_specification in cls._split_architecture(specification[3:-1]):
                residual_layers.extend(cls._create_layers(inner_specification))
            return [ResidualBlock(torch.nn.Sequential(*residual_layers))]

        if specification == "F":
            return [torch.nn.Flatten()]

        parts = specification.split("-")
        kind = parts[0]

        if kind in {"C", "CB"}:
            filters, kernel_size, stride, padding = parts[1:]
            modules: list[torch.nn.Module] = [
                torch.nn.LazyConv2d(
                    int(filters), kernel_size=int(kernel_size), stride=int(stride),
                    padding=cls._parse_padding(padding), bias=kind == "C",
                )
            ]
            if kind == "CB":
                modules.append(torch.nn.LazyBatchNorm2d())
            modules.append(torch.nn.ReLU())
            return modules

        if kind == "M":
            pool_size, stride = parts[1:]
            return [torch.nn.MaxPool2d(kernel_size=int(pool_size), stride=int(stride))]

        if kind == "H":
            hidden_layer_size = parts[1]
            return [torch.nn.LazyLinear(int(hidden_layer_size)), torch.nn.ReLU()]

        if kind == "D":
            return [torch.nn.Dropout(float(parts[1]))]

        raise ValueError(f"Unsupported layer specification '{specification}'.")

        # TODO: Note that you can construct a `TrainableModule` in two ways:
        # - either you create a `torch.nn.Module` (or a `torch.nn.Sequential` module) representing
        #   the whole network and pass it to the `super().__init__` call,
        # - or you start by calling `super().__init__()` without arguments and then assign the
        #   layers as attributes of `self`; in this case, you also need to implement the `forward`
        #   method that performs the forward pass through the model.


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it.
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev:")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
