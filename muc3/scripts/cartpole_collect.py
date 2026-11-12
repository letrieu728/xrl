"""Create a deterministic CartPole replay checkpoint for evaluation only."""

import json
from pathlib import Path

import gym
import numpy as np


SEED = 99
N_EPISODES = 200
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cartpole"


def policy(state):
    """The hand-written linear CartPole policy used by the repository notebook."""
    return int((state[0] / 2.4) * 0.4 + (state[2] / 2.095) * 0.6 >= 0)


def main():
    np.random.seed(SEED)
    env = gym.make("CartPole-v0")
    env.seed(SEED)
    states, actions, rewards, next_states = [], [], [], []
    episode_lengths = []

    for episode in range(N_EPISODES):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            # Match cxrl.lib.experience.play(): terminal transition is excluded.
            if not done:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
            state = next_state
            steps += 1
        episode_lengths.append(steps)
    env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "replay.npz",
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        next_states=np.asarray(next_states, dtype=float),
    )
    metadata = {
        "environment": "CartPole-v0",
        "seed": SEED,
        "episodes": N_EPISODES,
        "policy": "repository notebook hand-written linear rule",
        "transitions": len(states),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_min": int(np.min(episode_lengths)),
        "episode_length_max": int(np.max(episode_lengths)),
    }
    (OUT_DIR / "collection_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
