import os
import tempfile

from splendor_v1.env.env import SplendorEnv

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer
)

from splendor_v1.training_v2.replay_generator import (
    ReplayGenerator
)

from splendor_v1.training_v2.state_serializer import (
    serialize_state
)


# ============================================================
# TEST GENERATOR
# ============================================================

class TestReplayGenerator(
    ReplayGenerator
):
    """
    Minimal concrete ReplayGenerator.

    We don't need actual game generation yet.
    This only allows us to test the common
    ReplayGenerator functionality.
    """

    def generate_game(
        self,
        seed=None,
        **kwargs,
    ):
        raise NotImplementedError


# ============================================================
# SETUP
# ============================================================

env = SplendorEnv()

replay_buffer = ReplayBuffer(
    capacity=1000
)

generator = TestReplayGenerator(
    env=env,
    replay_buffer=replay_buffer,
    state_serializer=serialize_state,
)


# ============================================================
# RESET
# ============================================================

state = generator.reset(
    seed=12345
)

print()
print("=" * 70)
print("STATE")
print("=" * 70)

print(
    "current_player:",
    state.current_player
)

print(
    "turn_number:",
    state.turn_number
)

print(
    "node_type:",
    state.node_type
)


# ============================================================
# STATE SERIALIZATION
# ============================================================

serialized_state = (
    generator.serialize_state(
        state
    )
)

print()
print("=" * 70)
print("SERIALIZED STATE")
print("=" * 70)

print(
    "keys:",
    serialized_state.keys()
)

print(
    "current_player:",
    serialized_state[
        "current_player"
    ]
)

print(
    "turn_number:",
    serialized_state[
        "turn_number"
    ]
)

print(
    "node_type:",
    serialized_state[
        "node_type"
    ]
)

print(
    "bank:",
    serialized_state[
        "bank"
    ]
)

print(
    "visible cards:",
    serialized_state[
        "visible_card_ids"
    ]
)

print(
    "deck sizes:",
    {
        tier: len(cards)
        for tier, cards
        in serialized_state[
            "deck_card_ids"
        ].items()
    }
)


# ============================================================
# BUILD POSITION SAMPLE
# ============================================================

sample = generator.build_base_sample(
    state=state,
    step_index=0,
)

print()
print("=" * 70)
print("POSITION SAMPLE")
print("=" * 70)

print(
    "sample keys:"
)

for key in sample.keys():
    print("   ", key)


# ============================================================
# CHECK LEGAL ACTIONS
# ============================================================

legal_actions = sample[
    "legal_actions"
]

legal_action_ids = sample[
    "legal_action_ids"
]

print()
print(
    "legal actions:",
    len(legal_actions)
)

print(
    "legal action ids:",
    len(legal_action_ids)
)

assert (
    len(legal_actions)
    == len(legal_action_ids)
)

assert (
    len(legal_actions)
    > 0
)

print()
print(
    "first legal action ID:",
    legal_action_ids[0]
)

print(
    "first semantic action:",
    legal_actions[0]
)


# ============================================================
# CHOOSE ONE LEGAL ACTION
# ============================================================

raw_legal_actions = (
    generator.get_legal_actions(
        state
    )
)

action = raw_legal_actions[0]

generator.record_chosen_action(
    sample,
    action,
)

print()
print("=" * 70)
print("CHOSEN ACTION")
print("=" * 70)

print(
    "chosen ID:",
    sample[
        "chosen_action_id"
    ]
)

print(
    "chosen semantic action:",
    sample[
        "chosen_action"
    ]
)


# ============================================================
# VERIFY ACTION ID
# ============================================================

expected_id = (
    env.action_to_id(
        action
    )
)

assert (
    sample[
        "chosen_action_id"
    ]
    == expected_id
)

print()
print(
    "action ID round-trip check: PASS"
)


# ============================================================
# STEP ENVIRONMENT
# ============================================================

reward, terminated, info = (
    generator.step(
        action
    )
)

generator.record_transition(
    sample,
    reward,
    terminated,
)

print()
print("=" * 70)
print("TRANSITION")
print("=" * 70)

print(
    "reward:",
    sample["reward"]
)

print(
    "terminated:",
    sample[
        "terminated_after_action"
    ]
)


# ============================================================
# FAKE ONE-GAME TRAJECTORY
#
# We are NOT testing actual game completion yet.
#
# We only want to verify that:
#
#     sample
#         ↓
#     ReplayBuffer
#         ↓
#     save
#         ↓
#     load
#
# survives correctly.
# ============================================================

trajectory = [
    sample
]

game_metadata = {

    "seed":
        12345,

    "source":
        "replay_v2_test",

    "winner_ids":
        [],

    "final_scores":
        None,

    "num_positions":
        1,

    "action_space_version":
        generator.action_space_version,

    "action_space_size":
        generator.action_space_size,
}


# ============================================================
# COMMIT
# ============================================================

result = generator.commit_game(
    trajectory=trajectory,
    game_metadata=game_metadata,
)

print()
print("=" * 70)
print("COMMIT")
print("=" * 70)

print(
    result
)

assert len(
    replay_buffer
) == 1


# ============================================================
# SAVE / LOAD
# ============================================================

test_path = os.path.join(
    tempfile.gettempdir(),
    "splendor_replay_v2_test.pkl",
)

replay_buffer.save(
    test_path
)

loaded = ReplayBuffer.load(
    test_path
)

print()
print("=" * 70)
print("SAVE / LOAD")
print("=" * 70)

print(
    "original buffer size:",
    len(replay_buffer)
)

print(
    "loaded buffer size:",
    len(loaded)
)

assert len(
    loaded
) == 1


# ============================================================
# VERIFY LOADED SAMPLE
# ============================================================

loaded_sample = (
    loaded.buffer[0]
)

assert (
    loaded_sample[
        "chosen_action_id"
    ]
    ==
    sample[
        "chosen_action_id"
    ]
)

assert (
    loaded_sample[
        "chosen_action"
    ]
    ==
    sample[
        "chosen_action"
    ]
)

assert (
    loaded_sample[
        "state"
    ]
    ==
    sample[
        "state"
    ]
)

assert (
    loaded_sample[
        "legal_actions"
    ]
    ==
    sample[
        "legal_actions"
    ]
)


print()
print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)

print(
    "Replay V2 base infrastructure "
    "appears healthy."
)