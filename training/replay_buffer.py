from collections import deque
import random


class ReplayBuffer:

    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def add(
        self,
        observation,
        target_policy,
        target_value,
    ):
        self.buffer.append(
            (
                observation,
                target_policy,
                target_value,
            )
        )

    def sample(self, batch_size):
        return random.sample(
            self.buffer,
            batch_size,
        )

    def __len__(self):
        return len(self.buffer)