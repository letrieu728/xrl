#!/usr/bin/env python3
"""PongDuel single-agent wrapper + heuristic policies (CausalXRL step 5).

The wrapper and the strong/weak policies are copied VERBATIM from the
CausalXRL repo notebook cxrl/jupyter/examples/PongDuel.ipynb (cells 1
and 5), with exactly two additions, both marked [ADDED]:

* reset(seed=None): gym 0.21-style seeding (env.seed(seed) before
  reset) so training/eval episodes are reproducible. The notebook
  wrapper has no seeding.
* The module docstring + comments below document the representation.

Representation (notebook wrapper):
* State: flat 20-dim float32 = [agent1 obs (10), agent2 obs (10)];
  each agent obs = [agent_row/40, agent_col/30, ball_row/40,
  ball_col/30, ball-dir one-hot (6: NW, W, SW, SE, E, NE)]. The two
  agents see the SAME ball position/direction; only the agent
  position differs, so columns 2-9 duplicate columns 12-19.
* Action: scalar joint action in {0..8} = a1*3 + a2, where a1, a2 in
  {0 NOOP, 1 UP, 2 DOWN} are the two paddles' moves.
* Reward: reward_mode="agent1" (notebook cell 3 default): the scalar
  reward is agent1's reward. PongDuel rewards are SPARSE: 0 on every
  step except the step a round ends, when the agent whose opponent
  missed the ball gets +1 (ma-gym PongDuel, max_rounds=10). Episode
  agent1 reward is in {0..10}; agent1+agent2 rewards always sum to 10
  at termination, so reward_mode="sum" would be constant and useless
  for learning. Per-episode agent2 reward and rounds are recorded
  from info["original_reward_n"] for reporting.
* Done: all(done_n) — both agents finish together after 10 rounds.
* Two-agent handling: ONE policy controls BOTH paddles through the
  joint action. This is the repo's own single-agent reduction, kept
  as-is. CONSEQUENCE, STATED PLAINLY: maximising agent1's reward
  admits (and rewards) sabotaging agent2 — e.g. holding agent2's
  paddle away from the ball so agent1 scores. A DQN trained this way
  is a JOINT policy, not a competitive agent1 that beats an
  independent agent2. Any "success" must be read with this in mind.

Policies:
* heuristic_policy(state): strong agent1 (predictive, deadband 0.75)
  + weak agent2 (deadband 6.0, waits until the ball is close) — the
  repo notebook's joint policy, also used for Mục 3's PongDuel data.
* random_policy(state, rng): uniform joint action 0..8.
"""
import sys
from pathlib import Path

# ma-gym is installed --no-deps into an isolated target dir (plain
# `pip install ma-gym` fails: it tries to rebuild gym 0.21 from source,
# which modern setuptools rejects). The gym 0.21 install in the
# step3-scratch venv is reused.
_MA_GYM_TARGET = Path(__file__).resolve().parent / "ma-gym-target"
if str(_MA_GYM_TARGET) not in sys.path:
    sys.path.insert(0, str(_MA_GYM_TARGET))

import gym  # noqa: E402
import ma_gym  # noqa: E402,F401  (registers PongDuel-v0)
import numpy as np  # noqa: E402
from gym import spaces  # noqa: E402


class SingleAgentPongDuelWrapper(gym.Wrapper):
    """Verbatim from PongDuel.ipynb cell 1, plus [ADDED] seeding."""

    def __init__(self, env, reward_mode="agent1"):
        super().__init__(env)
        self.n_agents = 2
        self.n_actions_per_agent = 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        assert reward_mode in ["agent1", "sum", "competitive"]
        self.reward_mode = reward_mode

    def flatten_obs(self, obs_n):
        return np.concatenate(
            [np.asarray(obs_n[0], dtype=np.float32),
             np.asarray(obs_n[1], dtype=np.float32)])

    def decode_action(self, action):
        action = int(action)
        a1 = action // 3
        a2 = action % 3
        return [a1, a2]

    def encode_action(self, a1, a2):
        return int(a1) * 3 + int(a2)

    def reset(self, seed=None, **kwargs):  # [ADDED] seed parameter
        if seed is not None:
            self.env.seed(seed)
        obs_n = self.env.reset()
        return self.flatten_obs(obs_n)

    def step(self, action):
        action_n = self.decode_action(action)
        obs_n, reward_n, done_n, info = self.env.step(action_n)
        obs = self.flatten_obs(obs_n)
        if self.reward_mode == "agent1":
            reward = reward_n[0]
        elif self.reward_mode == "sum":
            reward = reward_n[0] + reward_n[1]
        elif self.reward_mode == "competitive":
            reward = reward_n[0] - reward_n[1]
        done = all(done_n)
        info = dict(info)
        info["original_action_n"] = action_n
        info["original_reward_n"] = reward_n
        info["original_done_n"] = done_n
        return obs, reward, done, info


def make_pongduel_env(reward_mode="agent1"):
    base_env = gym.make("PongDuel-v0")
    return SingleAgentPongDuelWrapper(base_env, reward_mode=reward_mode)


# --- heuristic policies: verbatim from PongDuel.ipynb cell 5 ---
GRID_H = 40
GRID_W = 30
BALL_DIRECTIONS = ["NW", "W", "SW", "SE", "E", "NE"]
DIR_TO_VEL = {
    "NW": (-1, -1), "W": (0, -1), "SW": (1, -1),
    "SE": (1, 1), "E": (0, 1), "NE": (-1, 1),
}


def reflect_row(row, low=1, high=38):
    span = high - low
    period = 2 * span
    x = (row - low) % period
    if x <= span:
        return low + x
    else:
        return high - (x - span)


def strong_agent_policy(obs):
    obs = np.asarray(obs, dtype=float)
    agent_r = obs[0] * GRID_H
    agent_c = obs[1] * GRID_W
    ball_r = obs[2] * GRID_H
    ball_c = obs[3] * GRID_W
    direction_index = int(np.argmax(obs[4:]))
    direction = BALL_DIRECTIONS[direction_index]
    dr, dc = DIR_TO_VEL[direction]
    is_left_agent = agent_c < GRID_W / 2
    if is_left_agent:
        moving_toward_me = dc < 0
        contact_col = agent_c + 1
    else:
        moving_toward_me = dc > 0
        contact_col = agent_c - 1
    if moving_toward_me and dc != 0:
        steps_until_contact = (contact_col - ball_c) / dc
        if steps_until_contact > 0:
            target_r = reflect_row(ball_r + dr * steps_until_contact)
        else:
            target_r = GRID_H / 2
    else:
        target_r = GRID_H / 2
    deadband = 0.75
    if target_r < agent_r - deadband:
        return 1
    elif target_r > agent_r + deadband:
        return 2
    else:
        return 0


def weak_agent_policy(obs):
    obs = np.asarray(obs, dtype=float)
    agent_r = obs[0] * GRID_H
    ball_r = obs[2] * GRID_H
    ball_c = obs[3] * GRID_W
    deadband = 6.0
    if ball_c < GRID_W * 0.55:
        return 0
    if ball_r < agent_r - deadband:
        return 1
    elif ball_r > agent_r + deadband:
        return 2
    else:
        return 0


def encode_joint_action(a1, a2):
    return int(a1) * 3 + int(a2)


def heuristic_policy(state):
    """Repo joint policy: strong agent1 + weak agent2. state: (20,)."""
    state = np.asarray(state, dtype=float)
    a1 = strong_agent_policy(state[:10])
    a2 = weak_agent_policy(state[10:])
    return encode_joint_action(a1, a2)
