#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_counts: dict[str, int] = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_counts[line] = data_counts.get(line, 0) + 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_points = list(data_counts.keys())
    data_distribution = np.array([data_counts[data_point] for data_point in data_points], dtype=np.float64)
    data_distribution /= np.sum(data_distribution)

    # TODO: Load model distribution, each line `string \t probability`.
    model_distribution_mapping: dict[str, float] = {}
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            data_point, probability = line.split("\t")
            model_distribution_mapping[data_point] = float(probability)

    # TODO: Create a NumPy array containing the model distribution.
    model_distribution = np.array(
        [model_distribution_mapping.get(data_point, 0.0) for data_point in data_points], dtype=np.float64)

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_distribution * np.log(data_distribution))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # the resulting crossentropy should be `np.inf`.
    if np.any(model_distribution == 0):
        crossentropy = np.inf
    else:
        crossentropy = -np.sum(data_distribution * np.log(model_distribution))

    # TODO: Compute KL-divergence D_KL(data distribution, model distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate.
    return float(entropy), float(crossentropy), float(kl_divergence)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print(f"Entropy: {entropy:.2f} nats")
    print(f"Crossentropy: {crossentropy:.2f} nats")
    print(f"KL divergence: {kl_divergence:.2f} nats")
