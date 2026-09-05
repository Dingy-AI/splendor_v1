import pickle
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample

        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []

        sample_size = min(
            batch_size,
            len(self.buffer)
        )

        indices = np.random.choice(
            len(self.buffer),
            size=sample_size,
            replace=False,
        )

        return [
            self.buffer[i]
            for i in indices
        ]

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "buffer": self.buffer,
                    "position": self.position,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        replay_buffer = cls(
            capacity=data["capacity"]
        )

        replay_buffer.buffer = data["buffer"]
        replay_buffer.position = data["position"]

        return replay_buffer