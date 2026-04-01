#!/usr/bin/env python3
import argparse
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--decode_on_demand", default=False, action="store_true", help="Decode images on demand.")
parser.add_argument("--decoder_channels", default=256, type=int, help="Channels in the decoder bottleneck.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout in the decoder.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--model_name", default="tf_efficientnetv2_b0.in1k", type=str, help="Timm model name.")
parser.add_argument("--patience", default=6, type=int, help="Early stopping patience.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="AdamW weight decay.")


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CAGS.Dataset, preprocessing, *, training: bool = False) -> None:
        super().__init__(dataset)
        self._preprocessing = preprocessing
        self._training = training

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = example["image"], example["mask"]
        if self._training and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        image = self._preprocessing(image)
        return image, mask


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(dropout) if dropout else torch.nn.Identity(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([inputs, skip], dim=1))


class Model(npfl138.TrainableModule):
    def __init__(self, encoder: torch.nn.Module, encoder_channels: list[int], args: argparse.Namespace) -> None:
        super().__init__()
        self.encoder = encoder
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(encoder_channels[-1], args.decoder_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(args.decoder_channels),
            torch.nn.ReLU(),
        )

        decoder_channels = [max(args.decoder_channels // (2 ** i), 32) for i in range(len(encoder_channels) - 1)]
        in_channels = args.decoder_channels
        blocks = []
        for skip_channels, out_channels in zip(reversed(encoder_channels[:-1]), decoder_channels):
            blocks.append(DecoderBlock(in_channels, skip_channels, out_channels, args.dropout))
            in_channels = out_channels
        self.decoder = torch.nn.ModuleList(blocks)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        x = self.stem(features[-1])
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            x = block(x, skip)
        x = F.interpolate(x, size=(CAGS.H, CAGS.W), mode="bilinear", align_corners=False)
        return self.head(x)

    def compute_loss(self, y_pred, y_true, *inputs):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        probabilities = torch.sigmoid(y_pred)
        intersection = (probabilities * y_true).sum(dim=(1, 2, 3))
        union = probabilities.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        return bce + dice.mean()


def write_predictions(model: Model, dataloader: torch.utils.data.DataLoader, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as predictions_file:
        for mask in model.predict(dataloader, data_with_labels=True, as_numpy=True, console=0):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.0, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=args.decode_on_demand)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    encoder = timm.create_model(args.model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3, 4))

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=encoder.pretrained_cfg["mean"], std=encoder.pretrained_cfg["std"]),
    ])

    train = Dataset(cags.train, preprocessing, training=True).dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers,
    )
    dev = Dataset(cags.dev, preprocessing).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )
    test = Dataset(cags.test, preprocessing).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )

    # Create the model and train it.
    model = Model(encoder, list(encoder.feature_info.channels()), args)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_iou = float("-inf")
    epochs_without_improvement = 0

    def keep_best_weights(model: Model, epoch: int, logs: dict[str, float]):
        nonlocal best_state_dict, best_iou, epochs_without_improvement

        if logs["dev:iou"] > best_iou:
            best_iou = logs["dev:iou"]
            best_state_dict = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                return npfl138.STOP_TRAINING

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train)))
    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"iou": CAGS.MaskIoUMetric(from_logits=True)},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best_weights])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Best dev IoU: {100 * best_iou:.2f}%", flush=True)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    write_predictions(model, dev, os.path.join(logdir, "cags_segmentation_dev.txt"))
    write_predictions(model, test, os.path.join(logdir, "cags_segmentation.txt"))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
