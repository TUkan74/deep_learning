#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl138
npfl138.require_version("2526.11")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--controller", default=[0.5, 1.0, 10.0, 1.5], type=float, nargs=4,
                    help="Linear controller weights for decoded x, x_dot, theta, theta_dot.")


class Agent:
    _TAU = 0.02

    def __init__(self, args: argparse.Namespace) -> None:
        self._weights = np.asarray(args.controller, dtype=np.float32)

    def _decode_frame(self, frame: np.ndarray) -> tuple[float, float]:
        cart_pixels = np.argwhere(frame == 128)
        if len(cart_pixels) == 0:
            return 0.0, 0.0

        cart = float(cart_pixels[:, 1].mean())
        x = (cart - 40.0) * 3.0 / 40.0

        pole_pixels = np.argwhere(frame > 200)
        if len(pole_pixels) == 0:
            return x, 0.0

        base = np.asarray([70.0, cart], dtype=np.float32)
        distances = np.linalg.norm(pole_pixels.astype(np.float32) - base, axis=1)
        tip = pole_pixels[distances >= distances.max() - 2].mean(axis=0)
        theta = np.arcsin(np.clip((float(tip[1]) - cart) / 56.0, -1.0, 1.0))
        return x, theta

    def _decode_observation(self, observation: np.ndarray) -> np.ndarray:
        x_old, theta_old = self._decode_frame(observation[:, :, 0])
        x, theta = self._decode_frame(observation[:, :, -1])
        x_dot = (x - x_old) / (2 * self._TAU)
        theta_dot = (theta - theta_old) / (2 * self._TAU)
        return np.asarray([x, x_dot, theta, theta_dot], dtype=np.float32)

    def predict(self, observation: np.ndarray) -> int:
        state = self._decode_observation(observation)
        return int(np.dot(self._weights, state) >= 0)


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    agent = Agent(args)

    # Final evaluation. The pixel observation is decoded back to the physical
    # CartPole state components visible in the rendered image.
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            action = agent.predict(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(
        gym.make("npfl138/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
