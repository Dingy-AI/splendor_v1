import numpy as np
import pytest

# Adjust these imports if your module paths differ.
from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
from splendor_v1.env.env import SplendorEnv


ACTION_SIZE = 1139
NUM_GAMES = 10


def _reset_env(env, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()

    if hasattr(result, "current_player"):
        return result

    if isinstance(result, tuple):
        for item in result:
            if hasattr(item, "current_player"):
                return item

    if hasattr(env, "state"):
        return env.state

    raise RuntimeError(
        "Could not locate GameState after env.reset(). "
        "Update _reset_env() to match your environment API."
    )


def _step_env(env, state, action):
    try:
        result = env.step(action)
    except TypeError:
        result = env.step(state, action)

    terminated = False

    if isinstance(result, tuple) and len(result) == 5:
        _, _, terminated, truncated, info = result
        terminated = bool(terminated or truncated)

        if hasattr(env, "state"):
            return env.state, terminated

        if isinstance(info, dict) and info.get("state") is not None:
            return info["state"], terminated

    if isinstance(result, tuple) and len(result) == 4:
        _, _, done, info = result
        terminated = bool(done)

        if hasattr(env, "state"):
            return env.state, terminated

        if isinstance(info, dict) and info.get("state") is not None:
            return info["state"], terminated

    if hasattr(result, "current_player"):
        next_state = result

        if hasattr(env, "_check_terminated"):
            terminated = bool(env._check_terminated(next_state))

        return next_state, terminated

    if hasattr(env, "state"):
        next_state = env.state

        if hasattr(env, "_check_terminated"):
            terminated = bool(env._check_terminated(next_state))

        return next_state, terminated

    raise RuntimeError(
        "Could not locate GameState after env.step(). "
        "Update _step_env() to match your environment API."
    )


def _assert_policy_is_valid(env, state, policy):
    assert isinstance(policy, np.ndarray)
    assert policy.shape == (ACTION_SIZE,)
    assert np.all(np.isfinite(policy))
    assert np.all(policy >= 0.0)

    legal_actions = env._legal_actions(state)

    if not legal_actions:
        assert np.isclose(policy.sum(), 0.0)
        return

    assert np.isclose(policy.sum(), 1.0, atol=1e-5), (
        f"Policy sum was {policy.sum()}"
    )

    legal_ids = {
        env.action_to_id(action)
        for action in legal_actions
    }

    nonzero_ids = set(
        np.flatnonzero(policy > 0.0).tolist()
    )

    illegal_nonzero = nonzero_ids - legal_ids

    assert not illegal_nonzero, (
        "Heuristic policy assigned probability to illegal "
        f"action ids: {sorted(illegal_nonzero)}"
    )


def test_heuristic_policy_on_initial_state():
    env = SplendorEnv()
    agent = HeuristicAgent3()

    state = _reset_env(env, seed=0)

    policy = agent.get_policy(
        env,
        state,
        action_size=ACTION_SIZE,
    )

    _assert_policy_is_valid(env, state, policy)

    action = agent.select_action(env, state)
    legal_actions = env._legal_actions(state)

    assert action is not None
    assert action in legal_actions

    print("\nInitial state")
    print("Legal actions:", len(legal_actions))
    print("Nonzero policy entries:", np.count_nonzero(policy))
    print("Max probability:", policy.max())
    print("Selected action:", action)


def test_selected_action_has_max_heuristic_score():
    env = SplendorEnv()
    agent = HeuristicAgent3()

    state = _reset_env(env, seed=1)

    scored_actions = agent._get_scored_candidates(
        env,
        state,
    )

    assert scored_actions

    selected_action = agent.select_action(
        env,
        state,
    )

    best_score = max(
        score
        for _, score in scored_actions
    )

    selected_score = next(
        score
        for action, score in scored_actions
        if action == selected_action
    )

    assert np.isclose(selected_score, best_score)


@pytest.mark.parametrize("seed", list(range(NUM_GAMES)))
def test_heuristic_agent_can_finish_games(seed):
    env = SplendorEnv()
    agent = HeuristicAgent3()

    state = _reset_env(env, seed=seed)

    terminated = False
    steps = 0
    max_steps = 500

    while not terminated:
        legal_actions = env._legal_actions(state)

        assert legal_actions, (
            f"Seed {seed}: non-terminal state had no legal actions."
        )

        policy = agent.get_policy(
            env,
            state,
            action_size=ACTION_SIZE,
        )

        _assert_policy_is_valid(
            env,
            state,
            policy,
        )

        action = agent.select_action(
            env,
            state,
        )

        assert action is not None
        assert action in legal_actions, (
            f"Seed {seed}: heuristic selected illegal action {action}"
        )

        state, terminated = _step_env(
            env,
            state,
            action,
        )

        steps += 1

        assert steps <= max_steps, (
            f"Seed {seed}: game exceeded {max_steps} steps."
        )

    print(f"\nSeed {seed}: completed in {steps} steps")


def test_print_sample_policies_from_game():
    env = SplendorEnv()
    agent = HeuristicAgent3()

    state = _reset_env(env, seed=123)

    terminated = False
    steps = 0
    samples_printed = 0

    while not terminated and samples_printed < 10:
        policy = agent.get_policy(
            env,
            state,
            action_size=ACTION_SIZE,
            temperature=10.0,
        )

        _assert_policy_is_valid(
            env,
            state,
            policy,
        )

        nonzero_ids = np.flatnonzero(
            policy > 0.0
        )

        ranked_ids = sorted(
            nonzero_ids,
            key=lambda action_id: policy[action_id],
            reverse=True,
        )

        print(f"\n--- Step {steps} ---")
        print("Current player:", state.current_player)
        print("Candidates:", len(ranked_ids))

        for action_id in ranked_ids[:10]:
            print(
                int(action_id),
                f"{policy[action_id]:.4f}",
                env.id_to_action(int(action_id)),
            )

        action = agent.select_action(
            env,
            state,
        )

        print("Selected:", action)

        state, terminated = _step_env(
            env,
            state,
            action,
        )

        steps += 1
        samples_printed += 1

def test_replay_values_follow_current_player():

    env = SplendorEnv()
    agent = HeuristicAgent3()

    state = _reset_env(
        env,
        seed=0,
    )

    trajectory = []

    terminated = False
    step = 0

    # ---------------------------------
    # PLAY COMPLETE GAME
    # ---------------------------------

    while not terminated:

        acting_player = state.current_player

        observation = env.observation_encoder.encoder(
            state
        )

        policy = agent.get_policy(
            env,
            state,
            action_size=1139,
        )

        action = agent.select_action(
            env,
            state,
        )

        trajectory.append({
            "step": step,
            "player": acting_player,
            "node_type": state.node_type,
            "action": action,
            "observation": observation.copy(),
            "policy": policy.copy(),
        })

        state, terminated = _step_env(
            env,
            state,
            action,
        )

        step += 1

        assert step < 500

    # ---------------------------------
    # GET WINNER
    # ---------------------------------

    winners = state.winners

    print("\nWinners:", winners)

    # ---------------------------------
    # ASSIGN VALUES
    # ---------------------------------

    replay_samples = []

    for sample in trajectory:

        player = sample["player"]

        value = (
            1.0
            if player in winners
            else -1.0
        )

        replay_samples.append(
            (
                sample["observation"],
                sample["policy"],
                value,
            )
        )

        sample["value"] = value

    # ---------------------------------
    # PRINT EVERYTHING
    # ---------------------------------

    print()
    print("=" * 100)
    print("FULL REPLAY VALUE TRACE")
    print("=" * 100)

    for sample in trajectory:

        print(
            f"Step {sample['step']:3d} | "
            f"Player {sample['player']} | "
            f"{str(sample['node_type']):20s} | "
            f"Value {sample['value']:+.1f}"
        )

        print(
            f"    {sample['action']}"
        )

    # ---------------------------------
    # VERIFY SAVED VALUES
    # ---------------------------------

    assert len(
        replay_samples
    ) == len(
        trajectory
    )

    for sample, replay_sample in zip(
        trajectory,
        replay_samples,
    ):

        _, _, saved_value = replay_sample

        expected_value = (
            1.0
            if sample["player"] in winners
            else -1.0
        )

        assert saved_value == expected_value