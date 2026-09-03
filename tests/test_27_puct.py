# tests/mcts/test_puct.py

import math
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.node import Node
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.mcts.neural_evaluator import neural_evaluate


def test_puct_score_prefers_higher_prior_when_values_equal():

    env = SplendorEnv()
    env.reset()

    parent = Node(
        state=env.state.clone(),
        visits=10,
    )

    child_high_prior = Node(
        state=env.state.clone(),
        parent=parent,
        visits=1,
        value=0.0,
        prior=0.8,
    )

    child_low_prior = Node(
        state=env.state.clone(),
        parent=parent,
        visits=1,
        value=0.0,
        prior=0.2,
    )

    mcts = MCTS(
        simulations=1,
        selection_type="puct",
        model=SplendorNetwork(),
    )

    root_player = parent.state.current_player

    high_score = mcts.puct_score(
        parent,
        child_high_prior,
        root_player,
    )

    low_score = mcts.puct_score(
        parent,
        child_low_prior,
        root_player,
    )

    assert high_score > low_score


def test_puct_score_prefers_better_value_when_prior_equal():

    env = SplendorEnv()
    env.reset()

    parent = Node(
        state=env.state.clone(),
        visits=10,
    )

    good_child = Node(
        state=env.state.clone(),
        parent=parent,
        visits=2,
        value=1.6,
        prior=0.5,
    )

    bad_child = Node(
        state=env.state.clone(),
        parent=parent,
        visits=2,
        value=0.2,
        prior=0.5,
    )

    mcts = MCTS(
        simulations=1,
        selection_type="puct",
        model=SplendorNetwork(),
    )

    root_player = parent.state.current_player

    good_score = mcts.puct_score(
        parent,
        good_child,
        root_player,
    )

    bad_score = mcts.puct_score(
        parent,
        bad_child,
        root_player,
    )

    assert good_score > bad_score

def test_puct_exploration_decreases_with_child_visits():

    env = SplendorEnv()
    env.reset()

    parent = Node(
        state=env.state.clone(),
        visits=20,
    )

    less_visited = Node(
        state=env.state.clone(),
        parent=parent,
        visits=1,
        value=0.0,
        prior=0.5,
    )

    more_visited = Node(
        state=env.state.clone(),
        parent=parent,
        visits=10,
        value=0.0,
        prior=0.5,
    )

    mcts = MCTS(
        simulations=1,
        selection_type="puct",
        model=SplendorNetwork(),
    )

    root_player = parent.state.current_player

    score_less = mcts.puct_score(
        parent,
        less_visited,
        root_player,
    )

    score_more = mcts.puct_score(
        parent,
        more_visited,
        root_player,
    )

    assert score_less > score_more

def test_puct_negates_exploitation_on_opponent_turn():

    env = SplendorEnv()
    env.reset()

    root_player = env.state.current_player

    parent_state = env.state.clone()
    parent_state.current_player = 1 - root_player

    parent = Node(
        state=parent_state,
        visits=10,
    )

    child = Node(
        state=env.state.clone(),
        parent=parent,
        visits=2,
        value=1.0,
        prior=0.0,
    )

    mcts = MCTS(
        simulations=1,
        selection_type="puct",
        model=SplendorNetwork(),
    )

    score = mcts.puct_score(
        parent,
        child,
        root_player,
    )

    assert score < 0

def test_select_with_puct_chooses_higher_score_child():

    env = SplendorEnv()
    env.reset()

    root_player = env.state.current_player

    parent = Node(
        state=env.state.clone(),
        visits=10,
        untried_actions=[],
        expanded=True
    )

    child_a = Node(
        state=env.state.clone(),
        parent=parent,
        visits=1,
        value=0.0,
        prior=0.8,
        untried_actions=[],
    )

    child_b = Node(
        state=env.state.clone(),
        parent=parent,
        visits=1,
        value=0.0,
        prior=0.2,
        untried_actions=[],
    )

    parent.children = [
        child_a,
        child_b,
    ]

    mcts = MCTS(
        simulations=1,
        selection_type="puct",
        model=SplendorNetwork(),
    )

    selected = mcts.select(
        env,
        parent,
        root_player,
    )

    assert selected is child_a

def test_puct_requires_model():

    try:
        MCTS(
            simulations=1,
            selection_type="puct",
            model=None,
        )

        assert False

    except ValueError:
        pass

def test_puct_expansion_assigns_prior():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(
            env.state
        ),
    )

    child = mcts.expand(
        env,
        root,
    )

    assert child is not None

    assert child.prior >= 0.0
    assert child.prior <= 1.0

def test_puct_child_prior_matches_network_policy():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(
            env.state
        ),
    )

    policy_probs, _ = neural_evaluate(
        env,
        model,
        root.state,
    )

    child = mcts.expand(
        env,
        root,
    )

    action_id = env.action_to_id(
        child.action
    )

    expected_prior = policy_probs[
        action_id
    ].item()

    assert math.isclose(
        child.prior,
        expected_prior,
        rel_tol=1e-6,
    )

def test_puct_mcts_search_returns_legal_action():

    env = SplendorEnv()
    env.reset()

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

    legal_actions = env._legal_actions(
        env.state
    )

    assert action in legal_actions

def test_puct_search_root_children_have_priors():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=10,
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

        assert 0.0 <= child.prior <= 1.0