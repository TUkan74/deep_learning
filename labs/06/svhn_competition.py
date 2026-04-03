#!/usr/bin/env python3
import argparse
import math
import os

import timm
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, sigmoid_focal_loss
import torchvision.transforms.v2 as v2

import bboxes_utils
import npfl138
npfl138.require_version("2526.6")
from npfl138.datasets.svhn import SVHN


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


# Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--anchor_ratios", default="0.6,1.0,1.6", type=str, help="Comma-separated anchor width/height ratios.")
parser.add_argument("--anchor_sizes", default="18,28,42,60", type=str, help="Comma-separated anchor sizes in resized-image pixels.")
parser.add_argument("--backbone_learning_rate", default=3e-5, type=float, help="Learning rate for the backbone during finetuning.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--box_beta", default=1 / 9, type=float, help="Smooth L1 beta for bbox regression.")
parser.add_argument("--box_loss_weight", default=1.0, type=float, help="Weight of the bbox regression loss.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--decode_on_demand", default=False, action="store_true", help="Decode images on demand.")
parser.add_argument("--epochs", default=3, type=int, help="Number of frozen-backbone epochs.")
parser.add_argument("--feature_level", default=2, type=int, help="EfficientNet feature level to use (0=112x112 ... 4=7x7).")
parser.add_argument("--finetune_epochs", default=12, type=int, help="Number of full finetuning epochs.")
parser.add_argument("--focal_alpha", default=0.25, type=float, help="Alpha parameter of focal loss.")
parser.add_argument("--focal_gamma", default=2.0, type=float, help="Gamma parameter of focal loss.")
parser.add_argument("--head_channels", default=256, type=int, help="Channels in the detector head.")
parser.add_argument("--image_size", default=224, type=int, help="Square image size used by the detector.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate for the detector head.")
parser.add_argument("--max_detections", default=6, type=int, help="Maximum number of detections per image after NMS.")
parser.add_argument("--model_name", default="tf_efficientnetv2_b0.in1k", type=str, help="Timm model name.")
parser.add_argument("--nms_threshold", default=0.35, type=float, help="IoU threshold used by NMS.")
parser.add_argument("--patience", default=4, type=int, help="Early stopping patience during finetuning.")
parser.add_argument("--pre_nms_topk", default=500, type=int, help="Maximum number of candidates per image before NMS.")
parser.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction, help="Use pretrained backbone weights when available.")
parser.add_argument("--score_threshold", default=0.25, type=float, help="Minimum detection score.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_iou_threshold", default=0.4, type=float, help="IoU threshold for assigning additional positive anchors.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="AdamW weight decay.")


class Dataset(npfl138.TransformedDataset):
    def __init__(
        self,
        dataset: SVHN.Dataset,
        image_size: tuple[int, int],
        preprocessing,
        interpolation: v2.InterpolationMode,
        *,
        training: bool = False,
    ) -> None:
        super().__init__(dataset)
        self._image_size = image_size
        self._preprocessing = preprocessing
        self._resize = v2.Resize(image_size, interpolation=interpolation)
        self._augmentation = v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03) \
            if training else torch.nn.Identity()

    def transform(self, example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = example["image"]
        classes = example["classes"].to(torch.int64)
        bboxes = example["bboxes"].to(torch.float32).clone()
        original_height, original_width = image.shape[-2:]

        scale_y = self._image_size[0] / original_height
        scale_x = self._image_size[1] / original_width
        bboxes[:, [bboxes_utils.TOP, bboxes_utils.BOTTOM]] *= scale_y
        bboxes[:, [bboxes_utils.LEFT, bboxes_utils.RIGHT]] *= scale_x

        image = self._augmentation(self._resize(image))
        image = self._preprocessing(image)
        return {
            "image": image,
            "classes": classes,
            "bboxes": bboxes,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.float32),
        }

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        images = torch.stack([example["image"] for example in batch])
        targets = [{key: value for key, value in example.items() if key != "image"} for example in batch]
        return images, targets


class Detector(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        in_channels: int,
        image_size: tuple[int, int],
        anchor_sizes: list[float],
        anchor_ratios: list[float],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self._backbone_trainable = True
        self.image_size = image_size
        self.num_classes = SVHN.LABELS
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_sizes) * len(anchor_ratios)

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, args.head_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(args.head_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(args.head_channels, args.head_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(args.head_channels),
            torch.nn.ReLU(),
        )
        self.objectness_head = torch.nn.Conv2d(args.head_channels, self.num_anchors, kernel_size=3, padding=1)
        self.classification_head = torch.nn.Conv2d(
            args.head_channels, self.num_anchors * self.num_classes, kernel_size=3, padding=1,
        )
        self.bbox_head = torch.nn.Conv2d(args.head_channels, self.num_anchors * 4, kernel_size=3, padding=1)

        with torch.no_grad():
            feature_map = self.backbone(torch.zeros(1, 3, *image_size))[0]
        self.feature_shape = tuple(feature_map.shape[-2:])
        self.register_buffer("anchors", self._create_anchors(self.feature_shape[0], self.feature_shape[1]))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self._backbone_trainable:
            self.backbone.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_trainable = trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        self.backbone.train(trainable)

    def _create_anchors(self, height: int, width: int) -> torch.Tensor:
        stride_y = self.image_size[0] / height
        stride_x = self.image_size[1] / width
        centers_y = (torch.arange(height, dtype=torch.float32) + 0.5) * stride_y
        centers_x = (torch.arange(width, dtype=torch.float32) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing="ij")

        anchors = []
        for anchor_size in self.anchor_sizes:
            for anchor_ratio in self.anchor_ratios:
                anchor_height = anchor_size / math.sqrt(anchor_ratio)
                anchor_width = anchor_size * math.sqrt(anchor_ratio)
                anchors.append(torch.stack([
                    grid_y - anchor_height / 2,
                    grid_x - anchor_width / 2,
                    grid_y + anchor_height / 2,
                    grid_x + anchor_width / 2,
                ], dim=-1))
        return torch.stack(anchors, dim=2).reshape(-1, 4)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(images)[0]
        features = self.stem(features)

        objectness_logits = self.objectness_head(features)
        classification_logits = self.classification_head(features)
        bbox_deltas = self.bbox_head(features)

        batch_size, _, feature_height, feature_width = objectness_logits.shape
        objectness_logits = objectness_logits.permute(0, 2, 3, 1).reshape(batch_size, -1)
        classification_logits = classification_logits.view(
            batch_size, self.num_anchors, self.num_classes, feature_height, feature_width,
        ).permute(0, 3, 4, 1, 2).reshape(batch_size, -1, self.num_classes)
        bbox_deltas = bbox_deltas.view(
            batch_size, self.num_anchors, 4, feature_height, feature_width,
        ).permute(0, 3, 4, 1, 2).reshape(batch_size, -1, 4)

        return objectness_logits, classification_logits, bbox_deltas

    def compute_loss(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        args: argparse.Namespace,
    ) -> dict[str, torch.Tensor]:
        objectness_logits, classification_logits, bbox_deltas = predictions
        anchors = self.anchors.to(objectness_logits.device)

        objectness_loss = objectness_logits.new_zeros(())
        classification_loss = objectness_logits.new_zeros(())
        bbox_loss = objectness_logits.new_zeros(())
        positive_anchors = 0

        for index, target in enumerate(targets):
            anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(
                anchors, target["classes"], target["bboxes"], args.train_iou_threshold,
            )
            positive = anchor_classes > 0
            objectness_loss = objectness_loss + sigmoid_focal_loss(
                objectness_logits[index], positive.to(objectness_logits.dtype),
                alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="sum",
            )

            if torch.any(positive):
                positive_anchors += int(positive.sum())
                classification_loss = classification_loss + F.cross_entropy(
                    classification_logits[index][positive], anchor_classes[positive] - 1, reduction="sum",
                )
                bbox_loss = bbox_loss + F.smooth_l1_loss(
                    bbox_deltas[index][positive], anchor_bboxes[positive],
                    beta=args.box_beta, reduction="sum",
                )

        normalizer = max(positive_anchors, 1)
        losses = {
            "objectness": objectness_loss / normalizer,
            "classification": classification_loss / normalizer,
            "bbox": bbox_loss / normalizer,
        }
        losses["loss"] = losses["objectness"] + losses["classification"] + args.box_loss_weight * losses["bbox"]
        return losses

    def detect(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        original_sizes: list[torch.Tensor],
        args: argparse.Namespace,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        objectness_logits, classification_logits, bbox_deltas = predictions
        anchors = self.anchors.to(objectness_logits.device)

        objectness_scores = objectness_logits.sigmoid()
        class_probabilities = classification_logits.softmax(dim=-1)
        class_scores, class_labels = class_probabilities.max(dim=-1)
        scores = objectness_scores * class_scores

        detections = []
        for index in range(len(original_sizes)):
            image_scores = scores[index]
            image_labels = class_labels[index]
            image_bboxes = bboxes_utils.bboxes_from_rcnn(anchors, bbox_deltas[index])

            image_bboxes = image_bboxes.clone()
            image_bboxes[:, [bboxes_utils.TOP, bboxes_utils.BOTTOM]] = image_bboxes[
                :, [bboxes_utils.TOP, bboxes_utils.BOTTOM]
            ].clamp(0, self.image_size[0])
            image_bboxes[:, [bboxes_utils.LEFT, bboxes_utils.RIGHT]] = image_bboxes[
                :, [bboxes_utils.LEFT, bboxes_utils.RIGHT]
            ].clamp(0, self.image_size[1])

            valid = (image_scores >= args.score_threshold) \
                & (image_bboxes[:, bboxes_utils.BOTTOM] > image_bboxes[:, bboxes_utils.TOP] + 1) \
                & (image_bboxes[:, bboxes_utils.RIGHT] > image_bboxes[:, bboxes_utils.LEFT] + 1)
            if not torch.any(valid):
                detections.append((
                    torch.empty([0], dtype=torch.int64),
                    torch.empty([0, 4], dtype=torch.float32),
                ))
                continue

            image_scores = image_scores[valid]
            image_labels = image_labels[valid]
            image_bboxes = image_bboxes[valid]

            if len(image_scores) > args.pre_nms_topk:
                best = torch.topk(image_scores, k=args.pre_nms_topk).indices
                image_scores = image_scores[best]
                image_labels = image_labels[best]
                image_bboxes = image_bboxes[best]

            keep = batched_nms(
                torch.stack([
                    image_bboxes[:, bboxes_utils.LEFT],
                    image_bboxes[:, bboxes_utils.TOP],
                    image_bboxes[:, bboxes_utils.RIGHT],
                    image_bboxes[:, bboxes_utils.BOTTOM],
                ], dim=-1),
                image_scores, image_labels, args.nms_threshold,
            )[:args.max_detections]

            image_labels = image_labels[keep].to(torch.int64).cpu()
            image_bboxes = image_bboxes[keep].cpu()
            scale_y = float(original_sizes[index][0]) / self.image_size[0]
            scale_x = float(original_sizes[index][1]) / self.image_size[1]
            image_bboxes[:, [bboxes_utils.TOP, bboxes_utils.BOTTOM]] *= scale_y
            image_bboxes[:, [bboxes_utils.LEFT, bboxes_utils.RIGHT]] *= scale_x
            detections.append((image_labels, image_bboxes))

        return detections


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]], device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def create_backbone(args: argparse.Namespace) -> tuple[torch.nn.Module, bool]:
    try:
        backbone = timm.create_model(
            args.model_name, pretrained=args.pretrained, features_only=True, out_indices=(args.feature_level,),
        )
        return backbone, args.pretrained
    except Exception as exception:
        if not args.pretrained:
            raise
        print(f"Pretrained backbone unavailable ({exception}); continuing without pretrained weights.", flush=True)
        backbone = timm.create_model(
            args.model_name, pretrained=False, features_only=True, out_indices=(args.feature_level,),
        )
        return backbone, False


def train_detector(
    model: Detector,
    train: torch.utils.data.DataLoader,
    dev: torch.utils.data.DataLoader,
    svhn: SVHN,
    args: argparse.Namespace,
) -> None:
    device = get_device()
    model.to(device)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_accuracy = float("-inf")
    epochs_without_improvement = 0

    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    backbone_parameters = list(model.backbone.parameters())

    def run_stage(epochs: int, optimizer: torch.optim.Optimizer, scheduler, *, backbone_trainable: bool) -> None:
        nonlocal best_state_dict, best_accuracy, epochs_without_improvement
        if epochs <= 0:
            return

        model.set_backbone_trainable(backbone_trainable)
        for epoch in range(epochs):
            model.train()
            epoch_losses = {"loss": 0.0, "objectness": 0.0, "classification": 0.0, "bbox": 0.0}
            for images, targets in train:
                images = images.to(device)
                targets = move_targets_to_device(targets, device)

                optimizer.zero_grad()
                predictions = model(images)
                losses = model.compute_loss(predictions, targets, args)
                losses["loss"].backward()
                optimizer.step()
                scheduler.step()

                for name in epoch_losses:
                    epoch_losses[name] += float(losses[name].detach())

            for name in epoch_losses:
                epoch_losses[name] /= max(len(train), 1)

            dev_predictions = predict(model, dev, args, device)
            dev_accuracy = SVHN.evaluate(svhn.dev, dev_predictions)
            print(
                f"Epoch {epoch + 1}/{epochs}"
                f": loss={epoch_losses['loss']:.4f}"
                f", obj={epoch_losses['objectness']:.4f}"
                f", cls={epoch_losses['classification']:.4f}"
                f", box={epoch_losses['bbox']:.4f}"
                f", dev accuracy={100 * dev_accuracy:.2f}%",
                flush=True,
            )

            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                best_state_dict = {
                    key: value.detach().to("cpu").clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if backbone_trainable and epochs_without_improvement >= args.patience:
                    return

    if args.epochs:
        head_optimizer = torch.optim.AdamW(head_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer, T_max=max(1, args.epochs * len(train)),
        )
        run_stage(args.epochs, head_optimizer, head_scheduler, backbone_trainable=False)

    if args.finetune_epochs:
        epochs_without_improvement = 0
        finetune_optimizer = torch.optim.AdamW([
            {"params": head_parameters, "lr": args.learning_rate},
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
        ], weight_decay=args.weight_decay)
        finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            finetune_optimizer, T_max=max(1, args.finetune_epochs * len(train)),
        )
        run_stage(args.finetune_epochs, finetune_optimizer, finetune_scheduler, backbone_trainable=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Best dev accuracy: {100 * best_accuracy:.2f}%", flush=True)


def predict(
    model: Detector,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[tuple[list[int], list[list[float]]]]:
    predictions: list[tuple[list[int], list[list[float]]]] = []
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            batch_predictions = model.detect(model(images.to(device)), [target["original_size"] for target in targets], args)
            for classes, bboxes in batch_predictions:
                predictions.append((classes.tolist(), bboxes.tolist()))
    return predictions


def write_predictions(
    path: str, predictions: list[tuple[list[int], list[list[float]]]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as predictions_file:
        for predicted_classes, predicted_bboxes in predictions:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + list(map(float, bbox))
            print(*output, file=predictions_file)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=args.decode_on_demand)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    backbone, pretrained_loaded = create_backbone(args)
    if not pretrained_loaded and args.epochs:
        print("Skipping frozen-backbone stage because the backbone is randomly initialized.", flush=True)
        args = argparse.Namespace(**({**vars(args), "epochs": 0}))

    image_size = (args.image_size, args.image_size)
    interpolation = v2.InterpolationMode(backbone.pretrained_cfg["interpolation"])

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=backbone.pretrained_cfg["mean"], std=backbone.pretrained_cfg["std"]),
    ])

    train = Dataset(
        svhn.train, image_size, preprocessing, interpolation, training=True,
    ).dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers,
    )
    dev = Dataset(
        svhn.dev, image_size, preprocessing, interpolation,
    ).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )
    test = Dataset(
        svhn.test, image_size, preprocessing, interpolation,
    ).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )

    # Create the model and train it.
    model = Detector(
        backbone,
        in_channels=backbone.feature_info.channels()[0],
        image_size=image_size,
        anchor_sizes=parse_csv_floats(args.anchor_sizes),
        anchor_ratios=parse_csv_floats(args.anchor_ratios),
        args=args,
    )
    train_detector(model, train, dev, svhn, args)

    device = get_device()
    model.to(device)
    dev_predictions = predict(model, dev, args, device)
    test_predictions = predict(model, test, args, device)
    print(f"Final dev accuracy: {100 * SVHN.evaluate(svhn.dev, dev_predictions):.2f}%", flush=True)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    write_predictions(os.path.join(logdir, "svhn_competition_dev.txt"), dev_predictions)
    write_predictions(os.path.join(logdir, "svhn_competition.txt"), test_predictions)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
