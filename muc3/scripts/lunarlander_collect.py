"""Create the 30-episode heuristic LunarLander replay used by evaluation notebooks."""

import json
from pathlib import Path

import gym
import numpy as np


SEED = 99
N_EPISODES = 30
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "lunarlander"


def policy(state):
    x, y, vx, vy, angle, angular_velocity, left_contact, right_contact = state
    target_angle = np.clip(0.5 * x + vx, -0.4, 0.4)
    target_y = 0.55 * abs(x)
    angle_control = 0.5 * (target_angle - angle) - angular_velocity
    hover_control = 0.5 * (target_y - y) - 0.5 * vy
    if left_contact or right_contact:
        angle_control = 0.0
        hover_control = -0.5 * vy
    if hover_control > abs(angle_control) and hover_control > 0.05:
        return 2
    if angle_control < -0.05:
        return 3
    if angle_control > 0.05:
        return 1
    return 0


def main():
    np.random.seed(SEED)
    env = gym.make("LunarLander-v2")
    env.seed(SEED)
    states, actions, rewards, next_states, lengths = [], [], [], [], []
    for _ in range(N_EPISODES):
        state = env.reset()
        done, steps = False, 0
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if not done:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
            state, steps = next_state, steps + 1
        lengths.append(steps)
    env.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "replay.npz",
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=int),
        rewards=np.asarray(rewards, dtype=float),
        next_states=np.asarray(next_states, dtype=float),
    )
    metadata = {
        "environment": "LunarLander-v2", "seed": SEED, "episodes": N_EPISODES,
        "policy": "repository notebook hand-written PD controller",
        "transitions": len(states), "episode_length_mean": float(np.mean(lengths)),
        "episode_length_min": int(np.min(lengths)), "episode_length_max": int(np.max(lengths)),
    }
    (OUT_DIR / "collection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
