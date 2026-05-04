#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import transformers

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate.")
parser.add_argument("--max_answer_length", default=30, type=int, help="Maximum decoded answer length in tokens.")
parser.add_argument("--max_length", default=384, type=int, help="Maximum sequence length.")
parser.add_argument("--n_best_size", default=20, type=int, help="Number of start/end candidates to consider.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stride", default=128, type=int, help="Context sliding-window stride.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup ratio.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="AdamW weight decay.")


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ReadingComprehensionDataset.Dataset, tokenizer: transformers.PreTrainedTokenizerBase,
                 max_length: int, stride: int) -> None:
        self._features = []
        for paragraph in dataset.paragraphs:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]
                answer_start = answer["start"]
                answer_end = answer_start + len(answer["text"])
                encoded = tokenizer(
                    question,
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    stride=stride,
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                )
                for i in range(len(encoded["input_ids"])):
                    sequence_ids = encoded.sequence_ids(i)
                    offsets = encoded["offset_mapping"][i]
                    context_tokens = [j for j, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
                    if not context_tokens:
                        continue
                    if offsets[context_tokens[0]][0] > answer_start or offsets[context_tokens[-1]][1] < answer_end:
                        continue

                    start_position = end_position = None
                    for token_index in context_tokens:
                        start, end = offsets[token_index]
                        if start <= answer_start < end:
                            start_position = token_index
                        if start < answer_end <= end:
                            end_position = token_index
                            break
                    if start_position is None or end_position is None:
                        continue

                    self._features.append({
                        "input_ids": encoded["input_ids"][i],
                        "attention_mask": encoded["attention_mask"][i],
                        "start_positions": start_position,
                        "end_positions": end_position,
                    })

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> dict:
        return self._features[index]


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ReadingComprehensionDataset.Dataset, tokenizer: transformers.PreTrainedTokenizerBase,
                 max_length: int, stride: int) -> None:
        self._features = []
        self._num_examples = 0
        for paragraph in dataset.paragraphs:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                example_id = self._num_examples
                self._num_examples += 1
                encoded = tokenizer(
                    qa["question"],
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    stride=stride,
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                )
                for i in range(len(encoded["input_ids"])):
                    sequence_ids = encoded.sequence_ids(i)
                    self._features.append({
                        "input_ids": encoded["input_ids"][i],
                        "attention_mask": encoded["attention_mask"][i],
                        "offset_mapping": [
                            offset if sequence_id == 1 else None
                            for offset, sequence_id in zip(encoded["offset_mapping"][i], sequence_ids)
                        ],
                        "context": context,
                        "example_id": example_id,
                    })

    @property
    def num_examples(self) -> int:
        return self._num_examples

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> dict:
        return self._features[index]


class Collator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase, with_labels: bool) -> None:
        self._tokenizer = tokenizer
        self._with_labels = with_labels

    def __call__(self, batch: list[dict]):
        features = [{key: item[key] for key in ["input_ids", "attention_mask"]} for item in batch]
        inputs = self._tokenizer.pad(features, padding="longest", return_tensors="pt")
        inputs = {key: value.long() for key, value in inputs.items()}
        if self._with_labels:
            labels = {
                "start_positions": torch.tensor([item["start_positions"] for item in batch], dtype=torch.long),
                "end_positions": torch.tensor([item["end_positions"] for item in batch], dtype=torch.long),
            }
            return inputs, labels
        return inputs, [{
            "offset_mapping": item["offset_mapping"],
            "context": item["context"],
            "example_id": item["example_id"],
        } for item in batch]


class Model(npfl138.TrainableModule):
    def __init__(self, robeczech: transformers.PreTrainedModel) -> None:
        super().__init__()
        self._robeczech = robeczech
        self._dropout = torch.nn.Dropout(robeczech.config.hidden_dropout_prob)
        self._qa_outputs = torch.nn.Linear(robeczech.config.hidden_size, 2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self._robeczech(**inputs)
        logits = self._qa_outputs(self._dropout(outputs.last_hidden_state))
        start_logits, end_logits = logits.split(1, dim=-1)
        return {"start_logits": start_logits.squeeze(-1), "end_logits": end_logits.squeeze(-1)}

    def compute_loss(self, y_pred, y, *xs):
        return (
            torch.nn.functional.cross_entropy(y_pred["start_logits"], y["start_positions"]) +
            torch.nn.functional.cross_entropy(y_pred["end_logits"], y["end_positions"])
        ) / 2


def predict_answers(model: Model, dataloader: torch.utils.data.DataLoader, num_examples: int,
                    max_answer_length: int, n_best_size: int) -> list[str]:
    best_answers = [("", -float("inf")) for _ in range(num_examples)]
    model.eval()
    with torch.no_grad():
        for inputs, metadata in dataloader:
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(inputs)
            start_logits = outputs["start_logits"].cpu()
            end_logits = outputs["end_logits"].cpu()
            for item_index, meta in enumerate(metadata):
                offsets = meta["offset_mapping"]
                context = meta["context"]
                example_id = meta["example_id"]
                starts = torch.topk(
                    start_logits[item_index], min(n_best_size, start_logits.shape[1])
                ).indices.tolist()
                ends = torch.topk(
                    end_logits[item_index], min(n_best_size, end_logits.shape[1])
                ).indices.tolist()
                for start in starts:
                    for end in ends:
                        if (
                            start >= len(offsets) or end >= len(offsets) or
                            offsets[start] is None or offsets[end] is None or
                            end < start or end - start + 1 > max_answer_length
                        ):
                            continue
                        score = start_logits[item_index, start].item() + end_logits[item_index, end].item()
                        if score > best_answers[example_id][1]:
                            best_answers[example_id] = (
                                context[offsets[start][0]:offsets[end][1]], score,
                            )
    return [answer for answer, _ in best_answers]


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data.
    dataset = ReadingComprehensionDataset()

    # TODO: Create the model and train it.
    train_dataset = TrainDataset(dataset.train, tokenizer, args.max_length, args.stride)
    dev_dataset = PredictDataset(dataset.dev, tokenizer, args.max_length, args.stride)
    test_dataset = PredictDataset(dataset.test, tokenizer, args.max_length, args.stride)

    train = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(tokenizer, with_labels=True),
    )
    dev = torch.utils.data.DataLoader(
        dev_dataset, batch_size=args.batch_size, collate_fn=Collator(tokenizer, with_labels=False),
    )
    test = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=Collator(tokenizer, with_labels=False),
    )

    model = Model(robeczech)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    model.configure(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
        loss=None,
        metrics={},
        logdir=logdir,
    )

    best_accuracy, best_state = -1.0, None
    for _ in range(args.epochs):
        model.fit(train, epochs=1)
        dev_predictions = predict_answers(
            model, dev, dev_dataset.num_examples, args.max_answer_length, args.n_best_size,
        )
        dev_accuracy = ReadingComprehensionDataset.evaluate(dataset.dev, dev_predictions)
        print(f"Dev accuracy: {100 * dev_accuracy:.2f}%")
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Using best dev accuracy: {100 * best_accuracy:.2f}%")

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        predictions = predict_answers(
            model, test, test_dataset.num_examples, args.max_answer_length, args.n_best_size,
        )

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
