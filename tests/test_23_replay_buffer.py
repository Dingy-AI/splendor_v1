# tests/training/test_replay_buffer.py

import numpy as np
import pytest

from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.env.core.constants import OBSERVATION_SIZE

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

def make_example(value=1.0):

    observation = np.random.randn(
        OBSERVATION_SIZE
    ).astype(np.float32)

    target_policy = np.zeros(
        ACTION_SPACE_SIZE,
        dtype=np.float32,
    )

    target_policy[0] = 1.0

    target_value = value

    return (
        observation,
        target_policy,
        target_value,
    )


def test_replay_buffer_starts_empty():

    buffer = ReplayBuffer()

    assert len(buffer) == 0


def test_add_single_example():

    buffer = ReplayBuffer()

    observation, target_policy, target_value = make_example()

    buffer.add(
        (observation,
        target_policy,
        target_value)
    )

    assert len(buffer) == 1


def test_added_example_preserves_data():

    buffer = ReplayBuffer()

    observation, target_policy, target_value = make_example(
        value=-1.0
    )

    buffer.add(
        (observation,
        target_policy,
        target_value)
    )

    stored_observation, stored_policy, stored_value = buffer.buffer[0]

    assert np.array_equal(
        stored_observation,
        observation,
    )

    assert np.array_equal(
        stored_policy,
        target_policy,
    )

    assert stored_value == target_value


def test_sample_returns_correct_batch_size():

    buffer = ReplayBuffer()

    for _ in range(20):

        observation, target_policy, target_value = make_example()

        buffer.add(
            (observation,
            target_policy,
            target_value)
        )

    batch = buffer.sample(
        batch_size=15
    )

    assert len(batch) == 15


def test_sample_examples_have_correct_structure():

    buffer = ReplayBuffer()

    for _ in range(10):

        observation, target_policy, target_value = make_example()

        buffer.add(
            (observation,
            target_policy,
            target_value)
        )

    batch = buffer.sample(
        batch_size=5
    )

    for observation, target_policy, target_value in batch:

        assert observation.shape == (
            OBSERVATION_SIZE,
        )

        assert target_policy.shape == (
            ACTION_SPACE_SIZE,
        )

        assert np.isclose(
            target_policy.sum(),
            1.0,
        )

        assert target_value in (
            -1.0,
            0.0,
            1.0,
        )


def test_buffer_respects_max_size():

    buffer = ReplayBuffer(
        capacity=5
    )

    for i in range(10):

        observation, target_policy, _ = make_example()

        buffer.add(
            (observation,
            target_policy,
            float(i))
        )

    assert len(buffer) == 5

def test_buffer_removes_oldest_examples_when_full():

    buffer = ReplayBuffer(
        capacity=3
    )

    for value in [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]:
        observation, target_policy, _ = make_example()

        buffer.add(
            (
                observation,
                target_policy,
                value,
            )
        )

    stored_values = [
        example[2]
        for example in buffer.buffer
    ]

    assert set(stored_values) == {
        2.0,
        3.0,
        4.0,
    }

def test_sample_does_not_remove_examples():

    buffer = ReplayBuffer()

    for _ in range(10):

        observation, target_policy, target_value = make_example()

        buffer.add(
            (observation,
            target_policy,
            target_value)
        )

    original_size = len(buffer)

    buffer.sample(
        batch_size=5
    )

    assert len(buffer) == original_size


def test_replay_buffer_save_and_load(tmp_path):

    buffer = ReplayBuffer(capacity=3)

    for value in [0.0, 1.0, 2.0]:
        observation, target_policy, _ = make_example()

        buffer.add(
            (
                observation,
                target_policy,
                value,
            )
        )

    path = tmp_path / "replay_buffer.pkl"

    buffer.save(path)

    loaded_buffer = ReplayBuffer.load(path)

    assert loaded_buffer.capacity == buffer.capacity
    assert loaded_buffer.position == buffer.position
    assert len(loaded_buffer) == len(buffer)

    for original, loaded in zip(
        buffer.buffer,
        loaded_buffer.buffer,
    ):
        np.testing.assert_array_equal(
            original[0],
            loaded[0],
        )

        np.testing.assert_array_equal(
            original[1],
            loaded[1],
        )

        assert original[2] == loaded[2]


def test_loaded_replay_buffer_continues_overwriting_oldest(tmp_path):

    buffer = ReplayBuffer(capacity=3)

    for value in [0.0, 1.0, 2.0, 3.0]:
        observation, target_policy, _ = make_example()

        buffer.add(
            (
                observation,
                target_policy,
                value,
            )
        )

    path = tmp_path / "replay_buffer.pkl"

    buffer.save(path)

    loaded_buffer = ReplayBuffer.load(path)

    observation, target_policy, _ = make_example()

    loaded_buffer.add(
        (
            observation,
            target_policy,
            4.0,
        )
    )

    stored_values = sorted(
        example[2]
        for example in loaded_buffer.buffer
    )

    assert stored_values == [
        2.0,
        3.0,
        4.0,
    ]