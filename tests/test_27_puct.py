"""PUCT child selection and neural-search integration tests."""

import pytest

from splendor_v1.env.core.enums import NodeType
from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.mcts.neural_evaluator import neural_evaluate
from splendor_v1.mcts.node import Node
from splendor_v1.network.model import SplendorNetwork


@pytest.fixture
def env():
    environment = SplendorEnv()
    environment.reset(seed=0)
    return environment


@pytest.fixture
def model():
    network = SplendorNetwork()
    network.eval()
    return network


@pytest.fixture
def mcts(model):
    return MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )


def make_parent(env, *, visits=10):
    return Node(
        state=env.state.clone(),
        visits=visits,
        untried_actions=[],
        expanded=True,
    )


def add_child(parent, *, visits=0, value=0.0, prior=0.0):
    child = Node(
        state=parent.state.clone(),
        parent=parent,
        visits=visits,
        value=value,
        prior=prior,
        untried_actions=[],
        expanded=False,
    )
    parent.children.append(child)
    return child


def test_select_puct_child_prefers_higher_prior_when_values_equal(env, mcts):
    parent = make_parent(env)
    add_child(parent, visits=1, prior=0.2)
    high_prior = add_child(parent, visits=1, prior=0.8)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is high_prior


def test_select_puct_child_prefers_better_value_when_priors_equal(env, mcts):
    parent = make_parent(env)
    add_child(parent, visits=2, value=0.2, prior=0.5)
    good_child = add_child(parent, visits=2, value=1.6, prior=0.5)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is good_child


def test_select_puct_child_uses_average_value_not_value_sum(env, mcts):
    parent = make_parent(env)
    add_child(parent, visits=10, value=4.0, prior=0.0)
    higher_average = add_child(parent, visits=1, value=0.6, prior=0.0)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is higher_average


def test_select_puct_child_exploration_decreases_with_child_visits(env, mcts):
    parent = make_parent(env, visits=20)
    add_child(parent, visits=10, prior=0.5)
    less_visited = add_child(parent, visits=1, prior=0.5)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is less_visited


@pytest.mark.parametrize("parent_visits", [0, 10])
def test_select_puct_child_handles_unvisited_children(env, mcts, parent_visits):
    parent = make_parent(env, visits=parent_visits)
    add_child(parent, visits=1, prior=0.5)
    unvisited = add_child(parent, visits=0, prior=0.5)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is unvisited


def test_select_puct_child_negates_exploitation_on_opponent_turn(env, mcts):
    root_player = env.state.current_player
    parent = make_parent(env)
    parent.state.current_player = 1 - root_player

    add_child(parent, visits=2, value=1.0, prior=0.0)
    good_for_opponent = add_child(parent, visits=2, value=-1.0, prior=0.0)

    selected = mcts.select_puct_child(parent, root_player)

    assert selected is good_for_opponent


@pytest.mark.parametrize(
    "node_type",
    [NodeType.OVERFLOW_DISCARD, NodeType.NOBLE_CLAIM],
)
def test_select_puct_child_keeps_root_perspective_on_same_player_decision(
    env, mcts, node_type
):
    root_player = env.state.current_player
    parent = make_parent(env)
    parent.state.node_type = node_type

    bad_child = add_child(parent, visits=2, value=-1.0, prior=0.0)
    good_child = add_child(parent, visits=2, value=1.0, prior=0.0)

    # The action completes this player's turn. Selection must still use
    # the parent player, even though the children belong to the opponent.
    for child in (bad_child, good_child):
        child.state.node_type = NodeType.MAIN_DECISION
        child.state.current_player = 1 - root_player

    selected = mcts.select_puct_child(parent, root_player)

    assert selected is good_child


def test_select_puct_child_keeps_first_child_when_scores_tie(env, mcts):
    parent = make_parent(env)
    first_child = add_child(parent, visits=2, value=1.0, prior=0.5)
    add_child(parent, visits=2, value=1.0, prior=0.5)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is first_child


def test_select_puct_child_handles_all_negative_scores(env, mcts):
    parent = make_parent(env)
    add_child(parent, visits=1, value=-0.8, prior=0.0)
    best_child = add_child(parent, visits=1, value=-0.2, prior=0.0)

    selected = mcts.select_puct_child(parent, parent.state.current_player)

    assert selected is best_child


def test_select_puct_child_respects_c_puct(env, mcts):
    parent = make_parent(env, visits=4)
    root_player = parent.state.current_player
    higher_value = add_child(parent, visits=1, value=0.8, prior=0.1)
    higher_prior = add_child(parent, visits=1, value=0.0, prior=0.9)

    assert mcts.select_puct_child(parent, root_player, c_puct=0) is higher_value
    assert mcts.select_puct_child(parent, root_player, c_puct=3) is higher_prior


def test_select_puct_child_recalculates_when_parent_visits_change(env, mcts):
    parent = make_parent(env, visits=1)
    root_player = parent.state.current_player
    higher_value = add_child(parent, visits=1, value=0.5, prior=0.1)
    higher_prior = add_child(parent, visits=1, value=0.0, prior=0.2)

    assert mcts.select_puct_child(parent, root_player) is higher_value

    # Shared calculations are per comparison, not permanently cached.
    parent.visits = 100

    assert mcts.select_puct_child(parent, root_player) is higher_prior


def test_select_puct_child_rejects_parent_without_children(env, mcts):
    parent = make_parent(env)

    with pytest.raises(ValueError):
        mcts.select_puct_child(parent, parent.state.current_player)


def test_select_with_puct_chooses_higher_score_child(env, mcts):
    parent = make_parent(env)
    add_child(parent, visits=1, prior=0.2)
    high_prior = add_child(parent, visits=1, prior=0.8)

    selected = mcts.select(env, parent, parent.state.current_player)

    assert selected is high_prior


def test_puct_requires_model():
    with pytest.raises(ValueError):
        MCTS(
            simulations=1,
            selection_type="puct",
            model=None,
        )


def test_puct_child_priors_match_network_policy(env, model, mcts):
    root = Node(state=env.state.clone())
    legal_actions = env._legal_actions(root.state)

    policy_probs, _ = neural_evaluate(
        env,
        model,
        root.state,
        legal_actions=legal_actions,
    )

    mcts.expand_all_with_priors(
        env,
        root,
        root_player=root.state.current_player,
        teacher_mode=False,
    )

    assert root.expanded
    assert len(root.children) == len(legal_actions)
    assert len(policy_probs) == len(legal_actions)

    for legal_action, expected_prior, child in zip(
        legal_actions, policy_probs, root.children
    ):
        assert child.action == legal_action
        assert child.prior == pytest.approx(
            expected_prior.item(), rel=1e-6, abs=1e-8
        )


def test_puct_mcts_search_returns_legal_action(env, mcts):
    mcts.simulations = 10

    action = mcts.search(env, env.state)

    assert action in env._legal_actions(env.state)


def test_puct_search_root_children_have_priors(env, mcts):
    mcts.simulations = 10

    _, root = mcts.search(env, env.state, return_root=True)

    assert root.children
    assert all(0.0 <= child.prior <= 1.0 for child in root.children)
    assert sum(child.prior for child in root.children) == pytest.approx(
        1.0, rel=1e-6, abs=1e-8
    )
