# tests/mcts/test_search_modes.py

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.network.model import SplendorNetwork


def test_ucb_search_returns_legal_action():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    mcts = MCTS(
        simulations=10,
        rollout_type="random",
        selection_type="ucb",
    )

    action = mcts.search(
        env,
        env.state,
    )

    assert action in legal_actions

def test_puct_search_returns_legal_action():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    action = mcts.search(
        env,
        env.state,
    )

    assert action in legal_actions

def test_ucb_root_visits_equal_simulations():

    env = SplendorEnv()
    env.reset()

    simulations = 10

    mcts = MCTS(
        simulations=simulations,
        rollout_type="random",
        selection_type="ucb",
    )

    action, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    assert root.visits == simulations

def test_puct_root_visits_equal_simulations():

    env = SplendorEnv()
    env.reset()

    simulations = 10

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    action, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    assert root.visits == simulations

def test_puct_expands_all_root_legal_actions():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    action, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    assert root.expanded is True

    assert len(root.children) == len(
        legal_actions
    )

def test_ucb_only_expands_one_child_per_simulation():

    env = SplendorEnv()
    env.reset()

    mcts = MCTS(
        simulations=1,
        rollout_type="random",
        selection_type="ucb",
    )

    action, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    assert len(root.children) == 1

def test_puct_root_children_have_valid_priors():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    _, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    assert len(root.children) > 0

    for child in root.children:

        assert child.prior >= 0.0
        assert child.prior <= 1.0

def test_puct_root_priors_sum_to_one():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    _, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    total_prior = sum(
        child.prior
        for child in root.children
    )

    assert abs(
        total_prior - 1.0
    ) < 1e-5

def test_puct_first_simulation_visits_one_child():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    _, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    visited_children = [
        child
        for child in root.children
        if child.visits > 0
    ]

    assert len(visited_children) == 1

    assert visited_children[0].visits == 1

def test_puct_child_visits_sum_to_root_visits():

    env = SplendorEnv()
    env.reset()

    simulations = 10

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    _, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    child_visits = sum(
        child.visits
        for child in root.children
    )

    assert child_visits == simulations