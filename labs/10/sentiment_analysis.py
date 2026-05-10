#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--max_length", default=256, type=int, help="Maximum sequence length.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, tokenizer: transformers.PreTrainedTokenizerBase,
                 max_length: int, with_labels: bool = True) -> None:
        super().__init__(dataset)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._with_labels = with_labels

    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        encoded = self._tokenizer(
            example["document"],
            truncation=True,
            max_length=self._max_length,
        )
        features = dict(encoded)
        if not self._with_labels:
            return features
        label = torch.tensor(self.dataset.label_vocab.index(example["label"]), dtype=torch.long)
        return features, label

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        if self._with_labels:
            features, labels = zip(*batch)
        else:
            features, labels = batch, None

        padded = self._tokenizer.pad(features, padding="longest", return_tensors="pt")
        padded = {key: value.long() for key, value in padded.items()}
        if labels is None:
            return padded
        return padded, torch.stack(labels)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        self._eleczech = eleczech
        self._dropout = torch.nn.Dropout(eleczech.config.hidden_dropout_prob)
        self._classifier = torch.nn.Linear(eleczech.config.hidden_size, len(dataset.label_vocab))

    # TODO: Implement the model computation.
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self._eleczech(**inputs)
        pooled = outputs.last_hidden_state[:, 0]
        return self._classifier(self._dropout(pooled))


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train = Dataset(facebook.train, tokenizer, args.max_length).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = Dataset(facebook.dev, tokenizer, args.max_length).dataloader(batch_size=args.batch_size)
    test = Dataset(facebook.test, tokenizer, args.max_length, with_labels=False).dataloader(batch_size=args.batch_size)

    # Create the model.
    model = Model(args, eleczech, facebook.train)

    # TODO: Configure and train the model
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(
            "multiclass", num_classes=len(facebook.train.label_vocab),
        )},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate dev and test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "sentiment_analysis_dev.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(dev, data_with_labels=True)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(document_logits.argmax().item()), file=predictions_file)

    with open(os.path.join(logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(document_logits.argmax().item()), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
