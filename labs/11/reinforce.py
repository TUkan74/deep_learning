#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl138
npfl138.require_version("2526.11")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--normalize_returns", default=True, action=argparse.BooleanOptionalAction,
                    help="Normalize returns in every training batch.")


class Agent:
    # Use an accelerator if available.
    device = npfl138.trainable_module.get_auto_device()

    def __init__(self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy. Note that the shape
        # of the observations is available in `env.observation_space.shape`
        # and the number of actions in `env.action_space.n`.
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)
        self._normalize_returns = args.normalize_returns

        # TODO: Define an optimizer; using `torch.optim.Adam` optimizer is a good default.
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.CrossEntropyLoss(reduction="none")

    # The `npfl138.rl_utils.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to construct the cross-entropy loss with
        # `reduction="none"` argument and then weight the losses of the individual
        # examples by the corresponding returns.
        self._policy.train()
        weights = returns
        if self._normalize_returns and len(weights) > 1:
            weights = (weights - weights.mean()) / (weights.std(unbiased=False) + 1e-8)
        loss = (self._loss(self._policy(states), actions) * weights).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Define the prediction method returning policy probabilities.
        self._policy.eval()
        with torch.no_grad():
            return torch.softmax(self._policy(states), dim=-1)


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Construct the agent.
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform an episode.
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                probabilities = agent.predict(state[np.newaxis])[0]
                action = np.random.choice(env.action_space.n, p=probabilities)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns by summing rewards.
            returns = []
            discounted_return = 0
            for reward in reversed(rewards):
                discounted_return = reward + args.gamma * discounted_return
                returns.append(discounted_return)
            returns = list(reversed(returns))

            # TODO: Append states, actions and returns to the training batch.
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # TODO: Train using the generated batch.
        agent.train(np.asarray(batch_states), np.asarray(batch_actions), np.asarray(batch_returns, dtype=np.float32))

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action.
            action = np.argmax(agent.predict(state[np.newaxis])[0])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
