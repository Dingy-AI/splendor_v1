from splendor_v1.mcts.node import Node
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.network.model import SplendorNetwork

from splendor_v1.env.env import SplendorEnv

def test_flip_tree_values_flips_entire_subtree():

    root = Node(
        state=None,
        value=4.0,
        visits=10,
    )

    child_1 = Node(
        state=None,
        parent=root,
        value=2.0,
        visits=5,
    )

    child_2 = Node(
        state=None,
        parent=root,
        value=-3.0,
        visits=7,
    )

    grandchild = Node(
        state=None,
        parent=child_1,
        value=1.5,
        visits=3,
    )

    root.children = [
        child_1,
        child_2,
    ]

    child_1.children = [
        grandchild,
    ]

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )


    mcts.flip_tree_values(root)

    assert root.value == -4.0
    assert child_1.value == -2.0
    assert child_2.value == 3.0
    assert grandchild.value == -1.5

    # Visits should not change.
    assert root.visits == 10
    assert child_1.visits == 5
    assert child_2.visits == 7
    assert grandchild.visits == 3

def test_tree_values_not_flipped_when_player_does_not_change():

    root = Node(
        state=None,
        value=3.0,
    )

    child = Node(
        state=None,
        parent=root,
        value=-1.0,
    )

    root.children = [child]

    old_player = 0
    new_player = 0

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    if new_player != old_player:
        mcts.flip_tree_values(root)

    assert root.value == 3.0
    assert child.value == -1.0

def test_tree_values_flip_when_player_changes():

    root = Node(
        state=None,
        value=3.0,
    )

    child = Node(
        state=None,
        parent=root,
        value=-1.0,
    )

    root.children = [child]

    old_player = 0
    new_player = 1


    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    if new_player != old_player:
        mcts.flip_tree_values(root)

    assert root.value == -3.0
    assert child.value == 1.0

def test_flip_tree_values_twice_restores_original_values():

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = Node(
        state=None,
        value=5.0,
    )

    child = Node(
        state=None,
        parent=root,
        value=-2.5,
    )

    grandchild = Node(
        state=None,
        parent=child,
        value=0.75,
    )

    root.children = [child]
    child.children = [grandchild]

    original_values = (
        root.value,
        child.value,
        grandchild.value,
    )

    mcts.flip_tree_values(root)
    mcts.flip_tree_values(root)

    assert (
        root.value,
        child.value,
        grandchild.value,
    ) == original_values


def test_search_reuses_existing_root():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    state = env.state
    old_player = state.current_player

    action, root = mcts.search(
        env,
        state,
        return_root=True,
    )

    # Find the node corresponding to the action
    # we're actually going to play.
    reused_root = next(
        child
        for child in root.children
        if child.action == action
    )

    previous_visits = reused_root.visits

    env.step(action)

    new_player = env.state.current_player

    reused_root.parent = None

    if new_player != old_player:
        mcts.flip_tree_values(
            reused_root
        )

    _, new_root = mcts.search(
        env,
        env.state,
        root=reused_root,
        return_root=True,
    )

    # Search reused the exact Node object.
    assert new_root is reused_root

    # It remains a root.
    assert new_root.parent is None

    # Previous search statistics weren't lost.
    assert new_root.visits >= previous_visits

