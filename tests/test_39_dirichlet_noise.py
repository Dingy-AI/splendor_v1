import numpy as np
import pytest

from splendor_v1.mcts.mcts import MCTS
from splendor_v1.mcts.node import Node
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.env.env import SplendorEnv


@pytest.fixture
def model():
    model = SplendorNetwork()
    model.eval()
    return model




@pytest.fixture
def env():
    env = SplendorEnv()
    env.reset()
    return env

def make_root(priors):
    root = Node(state=None)

    for prior in priors:
        child = Node(
            state=None,
            parent=root,
            action=None,
            prior=prior,
        )
        root.children.append(child)

    return root


def test_dirichlet_noise_preserves_probability_sum(model):
    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = make_root([
        0.25,
        0.25,
        0.25,
        0.25,
    ])

    mcts.add_dirichlet_noise(
        root,
        alpha=0.3,
        epsilon=0.25,
    )

    total = sum(
        child.prior
        for child in root.children
    )

    assert total == pytest.approx(
        1.0,
        abs=1e-6,
    )

def test_dirichlet_noise_changes_priors(model):
    np.random.seed(123)

    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = make_root([
        0.25,
        0.25,
        0.25,
        0.25,
    ])

    before = [
        child.prior
        for child in root.children
    ]

    mcts.add_dirichlet_noise(
        root,
        alpha=0.3,
        epsilon=0.25,
    )

    after = [
        child.prior
        for child in root.children
    ]

    assert after != pytest.approx(before)

def test_dirichlet_noise_uses_correct_mixture(monkeypatch, model):
    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = make_root([
        0.50,
        0.30,
        0.20,
    ])

    fake_noise = np.array([
        0.10,
        0.20,
        0.70,
    ])

    monkeypatch.setattr(
        np.random,
        "dirichlet",
        lambda alpha: fake_noise,
    )

    mcts.add_dirichlet_noise(
        root,
        alpha=0.3,
        epsilon=0.25,
    )

    expected = [
        0.75 * 0.50 + 0.25 * 0.10,
        0.75 * 0.30 + 0.25 * 0.20,
        0.75 * 0.20 + 0.25 * 0.70,
    ]

    actual = [
        child.prior
        for child in root.children
    ]

    assert actual == pytest.approx(
        expected,
        abs=1e-6,
    )

def test_dirichlet_noise_handles_root_without_children(model):
    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = Node(state=None)

    mcts.add_dirichlet_noise(root)

    assert root.children == []

def test_search_adds_root_noise_once(
    monkeypatch,
    env,
    model,
):
    env.reset()
    state = env.state

    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    calls = 0

    def fake_add_noise(root):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        mcts,
        "add_dirichlet_noise",
        fake_add_noise,
    )

    mcts.search(
        env,
        state,
        add_root_noise=True,
    )

    assert calls == 1

def test_search_does_not_add_noise_when_disabled(
    monkeypatch,
    env,
    model,
):
    env.reset()
    state = env.state

    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    calls = 0

    def fake_add_noise(root):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        mcts,
        "add_dirichlet_noise",
        fake_add_noise,
    )

    mcts.search(
        env,
        state,
        add_root_noise=False,
    )

    assert calls == 0

def test_search_adds_noise_once_to_reused_root(
    monkeypatch,
    env,
    model,
):
    env.reset()
    state = env.state

    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    # First search creates an expanded root.
    _, root = mcts.search(
        env,
        state,
        return_root=True,
        add_root_noise=False,
    )

    assert root.expanded
    assert root.children

    calls = 0

    def fake_add_noise(root):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        mcts,
        "add_dirichlet_noise",
        fake_add_noise,
    )

    # Search again using the existing root.
    mcts.search(
        env,
        state,
        root=root,
        add_root_noise=True,
    )

    assert calls == 1