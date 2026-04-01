#!/usr/bin/env python3
import argparse
import os

import timm
import torch
import torchmetrics
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--decode_on_demand", default=False, action="store_true", help="Decode images on demand.")
parser.add_argument("--dropout", default=0.3, type=float, help="Classifier dropout.")
parser.add_argument("--epochs", default=3, type=int, help="Number of frozen-backbone epochs.")
parser.add_argument("--finetune_epochs", default=12, type=int, help="Number of full finetuning epochs.")
parser.add_argument("--finetune_learning_rate", default=3e-5, type=float, help="Learning rate during finetuning.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=3e-3, type=float, help="Learning rate for classifier training.")
parser.add_argument("--model_name", default="tf_efficientnetv2_b0.in1k", type=str, help="Timm model name.")
parser.add_argument("--patience", default=5, type=int, help="Early stopping patience during finetuning.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="AdamW weight decay.")


class Dataset(npfl138.TransformedDataset):
    def __init__(
        self, dataset: CAGS.Dataset, preprocessing, interpolation: v2.InterpolationMode, *, training: bool = False,
    ) -> None:
        super().__init__(dataset)
        self._preprocessing = preprocessing
        self._augmentation = v2.Compose([
            v2.RandomResizedCrop((CAGS.H, CAGS.W), scale=(0.7, 1.0), interpolation=interpolation),
            v2.RandomHorizontalFlip(),
        ]) if training else torch.nn.Identity()

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._preprocessing(self._augmentation(example["image"]))
        return image, example["label"]


class Model(npfl138.TrainableModule):
    def __init__(self, backbone: torch.nn.Module, args: argparse.Namespace) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(self.backbone.num_features, CAGS.LABELS),
        )
        self._backbone_trainable = True

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_trainable = trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        self.backbone.train(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self._backbone_trainable:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.mean(dim=(-2, -1))
        return self.classifier(features)


def write_predictions(model: Model, dataloader: torch.utils.data.DataLoader, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(dataloader, data_with_labels=True, console=0):
            print(prediction.argmax().item(), file=predictions_file)


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

    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0)

    interpolation = v2.InterpolationMode(backbone.pretrained_cfg["interpolation"])

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=backbone.pretrained_cfg["mean"], std=backbone.pretrained_cfg["std"]),
    ])

    train = Dataset(cags.train, preprocessing, interpolation, training=True).dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers,
    )
    dev = Dataset(cags.dev, preprocessing, interpolation).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )
    test = Dataset(cags.test, preprocessing, interpolation).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )

    # Create the model and train it.
    model = Model(backbone, args)
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_accuracy = float("-inf")
    current_patience: int | None = None
    epochs_without_improvement = 0

    def keep_best_weights(model: Model, epoch: int, logs: dict[str, float]):
        nonlocal best_state_dict, best_accuracy, current_patience, epochs_without_improvement

        if logs["dev:accuracy"] > best_accuracy:
            best_accuracy = logs["dev:accuracy"]
            best_state_dict = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if current_patience is not None and epochs_without_improvement >= current_patience:
                return npfl138.STOP_TRAINING

    if args.epochs:
        model.set_backbone_trainable(False)
        head_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer, T_max=max(1, args.epochs * len(train)),
        )
        model.configure(
            optimizer=head_optimizer,
            scheduler=head_scheduler,
            loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
            logdir=logdir,
        )
        current_patience = None
        model.fit(train, dev=dev, epochs=args.epochs, callbacks=[keep_best_weights])

    if args.finetune_epochs:
        model.set_backbone_trainable(True)
        finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_learning_rate, weight_decay=args.weight_decay)
        finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            finetune_optimizer, T_max=max(1, args.finetune_epochs * len(train)),
        )
        model.configure(
            optimizer=finetune_optimizer,
            scheduler=finetune_scheduler,
            loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
            logdir=logdir,
        )
        current_patience = args.patience
        epochs_without_improvement = 0
        model.fit(train, dev=dev, epochs=args.finetune_epochs, callbacks=[keep_best_weights])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Best dev accuracy: {100 * best_accuracy:.2f}%", flush=True)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    write_predictions(model, dev, os.path.join(logdir, "cags_classification_dev.txt"))
    write_predictions(model, test, os.path.join(logdir, "cags_classification.txt"))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
