#!/usr/bin/env python3
import argparse
import os

import torch
import torchvision.transforms.v2 as v2
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import npfl138
npfl138.require_version("2526.6")
from npfl138.datasets.svhn import SVHN


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


parser = argparse.ArgumentParser()
parser.add_argument("--anchor_ratios", default="0.5,1.0,2.0", type=str, help="Comma-separated Faster R-CNN anchor ratios.")
parser.add_argument("--anchor_sizes", default="12,24,48,96,192", type=str, help="Comma-separated anchor sizes for FPN levels.")
parser.add_argument("--backbone_learning_rate", default=1e-4, type=float, help="Backbone learning rate during finetuning.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--decode_on_demand", default=False, action="store_true", help="Decode images on demand.")
parser.add_argument("--epochs", default=1, type=int, help="Number of frozen-backbone epochs.")
parser.add_argument("--finetune_epochs", default=8, type=int, help="Number of full finetuning epochs.")
parser.add_argument("--gradient_clip_norm", default=5.0, type=float, help="Gradient clipping norm.")
parser.add_argument("--image_size", default=320, type=int, help="Image size used by the detector transform.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Detector head learning rate.")
parser.add_argument("--max_detections", default=6, type=int, help="Maximum number of detections per image after tuning.")
parser.add_argument("--min_detections", default=1, type=int, help="Minimum number of detections kept per image.")
parser.add_argument("--model_name", default="fasterrcnn_mobilenet_v3_large_fpn", type=str, help="Torchvision detector name.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
parser.add_argument("--nms_threshold", default=0.4, type=float, help="NMS threshold used by the detector.")
parser.add_argument("--patience", default=4, type=int, help="Early stopping patience during finetuning.")
parser.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction, help="Use pretrained detector weights.")
parser.add_argument("--raw_detections_per_image", default=16, type=int, help="Number of raw detections kept before tuned filtering.")
parser.add_argument("--score_threshold", default=0.25, type=float, help="Fallback score threshold used before tuning.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of CPU threads to use.")
parser.add_argument("--tune_score_threshold", default=True, action=argparse.BooleanOptionalAction, help="Tune score threshold on the dev set.")
parser.add_argument("--tune_threshold_max", default=0.95, type=float, help="Maximum threshold considered during tuning.")
parser.add_argument("--tune_threshold_min", default=0.05, type=float, help="Minimum threshold considered during tuning.")
parser.add_argument("--tune_threshold_steps", default=19, type=int, help="Number of thresholds considered during tuning.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="SGD weight decay.")


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: SVHN.Dataset, *, training: bool = False) -> None:
        super().__init__(dataset)
        self._augmentation = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03) \
            if training else torch.nn.Identity()

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = self._augmentation(example["image"].to(torch.float32) / 255.0)
        boxes = example["bboxes"].to(torch.float32)
        boxes = boxes[:, [SVHN.LEFT, SVHN.TOP, SVHN.RIGHT, SVHN.BOTTOM]]
        target = {
            "boxes": boxes,
            "labels": example["classes"].to(torch.int64) + 1,
        }
        return image, target

    def collate(
        self, batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        images, targets = zip(*batch)
        return list(images), list(targets)


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


def create_model(args: argparse.Namespace) -> tuple[torch.nn.Module, bool]:
    anchor_sizes = tuple((size,) for size in parse_csv_ints(args.anchor_sizes))
    anchor_ratios = tuple(parse_csv_floats(args.anchor_ratios))

    if args.model_name == "fasterrcnn_resnet50_fpn_v2":
        detector = fasterrcnn_resnet50_fpn_v2
        weights_enum = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    elif args.model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        detector = fasterrcnn_mobilenet_v3_large_fpn
        weights_enum = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported detector '{args.model_name}'.")

    pretrained_loaded = args.pretrained
    detector_kwargs = {
        "weights": weights_enum if args.pretrained else None,
        "min_size": args.image_size,
        "max_size": args.image_size,
        "box_score_thresh": 0.0,
        "box_nms_thresh": args.nms_threshold,
        "box_detections_per_img": args.raw_detections_per_image,
    }
    try:
        model = detector(**detector_kwargs)
    except Exception as exception:
        if not args.pretrained:
            raise
        print(f"Pretrained detector unavailable ({exception}); continuing without pretrained weights.", flush=True)
        detector_kwargs["weights"] = None
        model = detector(**detector_kwargs)
        pretrained_loaded = False

    expected_anchors = model.rpn.anchor_generator.num_anchors_per_location()
    if len(set(expected_anchors)) != 1:
        raise ValueError(f"Unsupported detector anchor layout {expected_anchors}.")

    if expected_anchors[0] % len(anchor_ratios):
        raise ValueError(
            f"Detector expects {expected_anchors[0]} anchors/location, which is incompatible "
            f"with {len(anchor_ratios)} ratios."
        )

    sizes_per_level = expected_anchors[0] // len(anchor_ratios)
    if sizes_per_level == 1:
        if len(anchor_sizes) != len(expected_anchors):
            raise ValueError(
                f"Detector expects {len(expected_anchors)} anchor sizes, got {len(anchor_sizes)}."
            )
        compatible_sizes = anchor_sizes
    else:
        if len(anchor_sizes) != sizes_per_level:
            raise ValueError(
                f"Detector expects {sizes_per_level} anchor sizes per FPN level, got {len(anchor_sizes)}."
            )
        compatible_sizes = tuple(tuple(size for (size,) in anchor_sizes) for _ in expected_anchors)

    anchor_generator = AnchorGenerator(
        sizes=compatible_sizes, aspect_ratios=(anchor_ratios,) * len(expected_anchors),
    )
    if anchor_generator.num_anchors_per_location() != expected_anchors:
        raise ValueError(
            "Custom anchors must preserve the detector's anchors-per-location layout. "
            f"Requested {anchor_generator.num_anchors_per_location()}, "
            f"expected {expected_anchors}."
        )
    model.rpn.anchor_generator = anchor_generator

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, SVHN.LABELS + 1)
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = args.nms_threshold
    model.roi_heads.detections_per_img = args.raw_detections_per_image
    return model, pretrained_loaded


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(trainable)


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    backbone_trainable: bool,
) -> torch.optim.Optimizer:
    if backbone_trainable:
        head_parameters = [
            parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("backbone.")
        ]
        backbone_parameters = [
            parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("backbone.")
        ]
        parameter_groups = [{"params": head_parameters, "lr": args.learning_rate}]
        if backbone_parameters:
            parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_learning_rate})
    else:
        parameter_groups = [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad]}]

    return torch.optim.SGD(
        parameter_groups,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def outputs_to_raw_predictions(outputs: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    raw_predictions = []
    for output in outputs:
        boxes = output["boxes"].detach().cpu()
        if len(boxes):
            boxes = boxes[:, [1, 0, 3, 2]]
        raw_predictions.append({
            "labels": (output["labels"].detach().cpu().to(torch.int64) - 1),
            "boxes": boxes.to(torch.float32),
            "scores": output["scores"].detach().cpu().to(torch.float32),
        })
    return raw_predictions


def format_predictions(
    raw_predictions: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
    max_detections: int,
    min_detections: int,
) -> list[tuple[list[int], list[list[float]]]]:
    predictions: list[tuple[list[int], list[list[float]]]] = []
    for prediction in raw_predictions:
        labels = prediction["labels"]
        boxes = prediction["boxes"]
        scores = prediction["scores"]

        if len(scores):
            order = torch.argsort(scores, descending=True)
            labels, boxes, scores = labels[order], boxes[order], scores[order]

            keep = scores >= score_threshold
            if torch.any(keep):
                labels, boxes, scores = labels[keep], boxes[keep], scores[keep]

            keep_count = min(max_detections, len(scores))
            if keep_count:
                labels, boxes = labels[:keep_count], boxes[:keep_count]
            else:
                labels = labels[:0]
                boxes = boxes[:0]

            if len(labels) < min_detections:
                fallback = min(min_detections, len(prediction["scores"]))
                labels = prediction["labels"][:fallback]
                boxes = prediction["boxes"][:fallback]
        predictions.append((labels.tolist(), boxes.tolist()))
    return predictions


@torch.no_grad()
def predict_raw(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    model.eval()
    raw_predictions: list[dict[str, torch.Tensor]] = []
    for images, _ in dataloader:
        outputs = model([image.to(device) for image in images])
        raw_predictions.extend(outputs_to_raw_predictions(outputs))
    return raw_predictions


def evaluate_predictions(svhn: SVHN, predictions: list[tuple[list[int], list[list[float]]]]) -> float:
    return SVHN.evaluate(svhn.dev, predictions)


def tune_postprocessing(
    raw_predictions: list[dict[str, torch.Tensor]],
    svhn: SVHN,
    args: argparse.Namespace,
) -> tuple[float, int, float]:
    if not args.tune_score_threshold:
        predictions = format_predictions(
            raw_predictions,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            min_detections=args.min_detections,
        )
        return args.score_threshold, args.max_detections, evaluate_predictions(svhn, predictions)

    thresholds = torch.linspace(args.tune_threshold_min, args.tune_threshold_max, steps=args.tune_threshold_steps).tolist()
    best_score = float("-inf")
    best_threshold = args.score_threshold
    best_max_detections = args.max_detections

    for max_detections in range(max(args.min_detections, 1), args.max_detections + 1):
        for score_threshold in thresholds:
            predictions = format_predictions(
                raw_predictions,
                score_threshold=float(score_threshold),
                max_detections=max_detections,
                min_detections=args.min_detections,
            )
            accuracy = evaluate_predictions(svhn, predictions)
            if accuracy > best_score:
                best_score = accuracy
                best_threshold = float(score_threshold)
                best_max_detections = max_detections

    return best_threshold, best_max_detections, best_score


def train_detector(
    model: torch.nn.Module,
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

    def run_stage(epochs: int, *, backbone_trainable: bool) -> None:
        nonlocal best_state_dict, best_accuracy, epochs_without_improvement
        if epochs <= 0:
            return

        set_backbone_trainable(model, backbone_trainable)
        optimizer = build_optimizer(model, args, backbone_trainable=backbone_trainable)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(train)))

        for epoch in range(epochs):
            model.train()
            if not backbone_trainable:
                model.backbone.eval()
            epoch_losses = {
                "loss": 0.0,
                "loss_classifier": 0.0,
                "loss_box_reg": 0.0,
                "loss_objectness": 0.0,
                "loss_rpn_box_reg": 0.0,
            }

            for images, targets in train:
                images = [image.to(device) for image in images]
                targets = move_targets_to_device(targets, device)

                optimizer.zero_grad()
                losses = model(images, targets)
                loss = sum(losses.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
                scheduler.step()

                epoch_losses["loss"] += float(loss.detach())
                for name in epoch_losses:
                    if name != "loss":
                        epoch_losses[name] += float(losses[name].detach())

            for name in epoch_losses:
                epoch_losses[name] /= max(len(train), 1)

            raw_dev_predictions = predict_raw(model, dev, device)
            dev_predictions = format_predictions(
                raw_dev_predictions,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                min_detections=args.min_detections,
            )
            dev_accuracy = evaluate_predictions(svhn, dev_predictions)
            print(
                f"Epoch {epoch + 1}/{epochs}"
                f": loss={epoch_losses['loss']:.4f}"
                f", cls={epoch_losses['loss_classifier']:.4f}"
                f", box={epoch_losses['loss_box_reg']:.4f}"
                f", obj={epoch_losses['loss_objectness']:.4f}"
                f", rpn_box={epoch_losses['loss_rpn_box_reg']:.4f}"
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
        run_stage(args.epochs, backbone_trainable=False)

    if args.finetune_epochs:
        epochs_without_improvement = 0
        run_stage(args.finetune_epochs, backbone_trainable=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Best dev accuracy before postprocessing tuning: {100 * best_accuracy:.2f}%", flush=True)


def write_predictions(path: str, predictions: list[tuple[list[int], list[list[float]]]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as predictions_file:
        for predicted_classes, predicted_bboxes in predictions:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + list(map(float, bbox))
            print(*output, file=predictions_file)


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}", **vars(args))

    svhn = SVHN(decode_on_demand=args.decode_on_demand)

    train = Dataset(svhn.train, training=True).dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers,
    )
    dev = Dataset(svhn.dev).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )
    test = Dataset(svhn.test).dataloader(
        batch_size=args.batch_size, num_workers=args.dataloader_workers,
    )

    model, pretrained_loaded = create_model(args)
    if not pretrained_loaded and args.epochs:
        print("Skipping frozen-backbone stage because pretrained detector weights are unavailable.", flush=True)
        args = argparse.Namespace(**{**vars(args), "epochs": 0})

    train_detector(model, train, dev, svhn, args)

    device = get_device()
    model.to(device)
    raw_dev_predictions = predict_raw(model, dev, device)
    tuned_threshold, tuned_max_detections, tuned_dev_accuracy = tune_postprocessing(raw_dev_predictions, svhn, args)
    print(
        f"Tuned score threshold={tuned_threshold:.2f}, max_detections={tuned_max_detections},"
        f" dev accuracy={100 * tuned_dev_accuracy:.2f}%",
        flush=True,
    )

    dev_predictions = format_predictions(
        raw_dev_predictions,
        score_threshold=tuned_threshold,
        max_detections=tuned_max_detections,
        min_detections=args.min_detections,
    )
    raw_test_predictions = predict_raw(model, test, device)
    test_predictions = format_predictions(
        raw_test_predictions,
        score_threshold=tuned_threshold,
        max_detections=tuned_max_detections,
        min_detections=args.min_detections,
    )

    print(f"Final dev accuracy: {100 * evaluate_predictions(svhn, dev_predictions):.2f}%", flush=True)
    write_predictions(os.path.join(logdir, "svhn_competition_dev.txt"), dev_predictions)
    write_predictions(os.path.join(logdir, "svhn_competition.txt"), test_predictions)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
