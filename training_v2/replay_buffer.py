import os
import pickle
from collections import Counter

import numpy as np


class ReplayBuffer:

    FORMAT_VERSION = 3
    SAMPLE_VERSION = 3

    def __init__(
        self,
        capacity=500_000,
        metadata=None,
    ):
        self.capacity = capacity

        # Circular position buffer.
        #
        # Each entry is now a dictionary rather than:
        #
        #     (observation, policy, value)
        #
        self.buffer = []
        self.position = 0

        # Game-level information.
        #
        # {
        #     game_id: {
        #         ...
        #     }
        # }
        self.games = {}

        # Number of samples from each game currently alive
        # in the circular replay buffer.
        self.game_sample_counts = {}

        self.next_game_id = 0

        # Dataset / run-level metadata.
        self.metadata = (
            {} if metadata is None
            else dict(metadata)
        )

    # ============================================================
    # INTERNAL GAME ACCOUNTING
    # ============================================================

    def _increment_game_count(self, game_id):

        if game_id is None:
            return

        self.game_sample_counts[game_id] = (
            self.game_sample_counts.get(
                game_id,
                0,
            )
            + 1
        )

    def _decrement_game_count(self, game_id):

        if game_id is None:
            return

        count = self.game_sample_counts.get(
            game_id,
            0,
        )

        if count <= 1:

            self.game_sample_counts.pop(
                game_id,
                None,
            )

            # No samples from this game remain in the
            # active replay buffer.
            self.games.pop(
                game_id,
                None,
            )

        else:

            self.game_sample_counts[game_id] = (
                count - 1
            )

    # ============================================================
    # ADD ONE SAMPLE
    # ============================================================

    def add(self, sample):

        if not isinstance(sample, dict):
            raise TypeError(
                "ReplayBuffer v3 expects each sample "
                "to be a dictionary."
            )

        sample = dict(sample)

        sample.setdefault(
            "sample_version",
            self.SAMPLE_VERSION,
        )

        # --------------------------------------------------------
        # Circular overwrite
        # --------------------------------------------------------

        if len(self.buffer) < self.capacity:

            self.buffer.append(
                sample
            )

        else:

            old_sample = (
                self.buffer[self.position]
            )

            if isinstance(
                old_sample,
                dict,
            ):
                self._decrement_game_count(
                    old_sample.get(
                        "game_id"
                    )
                )

            self.buffer[self.position] = (
                sample
            )

        # --------------------------------------------------------
        # Track surviving samples by game
        # --------------------------------------------------------

        self._increment_game_count(
            sample.get("game_id")
        )

        self.position = (
            self.position + 1
        ) % self.capacity

    # ============================================================
    # ADD COMPLETE GAME
    # ============================================================

    def add_game(
        self,
        samples,
        game_metadata,
    ):
        """
        Add an entire completed trajectory.

        samples:
            list[dict]

        game_metadata:
            dictionary containing game-level facts such as:

                seed
                source
                split
                winner_ids
                final_scores
                model_generation
                model_checkpoint
                search configuration
                rules_version
                encoder_version
                action_space_version
        """

        if not samples:
            raise ValueError(
                "Cannot add an empty game."
            )

        if len(samples) > self.capacity:
            raise ValueError(
                "A single game contains more samples "
                "than the entire replay-buffer capacity."
            )

        game_metadata = dict(
            game_metadata
        )

        # Caller may explicitly supply a game ID.
        game_id = game_metadata.get(
            "game_id"
        )

        if game_id is None:

            game_id = self.next_game_id

        self.next_game_id = max(
            self.next_game_id,
            game_id + 1,
        )

        # --------------------------------------------------------
        # Store game information
        # --------------------------------------------------------

        game_metadata["game_id"] = (
            game_id
        )

        game_metadata["num_positions"] = (
            len(samples)
        )

        self.games[game_id] = (
            game_metadata
        )

        # --------------------------------------------------------
        # Store individual positions
        # --------------------------------------------------------

        for step_index, sample in enumerate(
            samples
        ):

            sample = dict(sample)

            sample["game_id"] = (
                game_id
            )

            sample.setdefault(
                "step_index",
                step_index,
            )

            self.add(
                sample
            )

        return game_id

    # ============================================================
    # SAMPLE
    # ============================================================

    def sample(
        self,
        batch_size,
        split=None,
        replace=True,
    ):

        if len(self.buffer) == 0:
            return []

        # --------------------------------------------------------
        # Optional train / val filtering
        # --------------------------------------------------------

        if split is None:

            candidate_indices = (
                np.arange(
                    len(self.buffer)
                )
            )

        else:

            candidate_indices = []

            for i, sample in enumerate(
                self.buffer
            ):

                game_id = sample.get(
                    "game_id"
                )

                game = self.games.get(
                    game_id,
                    {}
                )

                if game.get("split") == split:

                    candidate_indices.append(
                        i
                    )

            candidate_indices = np.asarray(
                candidate_indices,
                dtype=np.int64,
            )

        if len(candidate_indices) == 0:
            return []

        sample_size = min(
            batch_size,
            len(candidate_indices),
        )

        # If replacement is enabled, retain the
        # old ReplayBuffer behavior.
        if replace:
            sample_size = batch_size

        selected = np.random.choice(
            candidate_indices,
            size=sample_size,
            replace=replace,
        )

        return [
            self.buffer[i]
            for i in selected
        ]

    # ============================================================
    # GET GAME TRAJECTORY
    # ============================================================

    def get_game_samples(
        self,
        game_id,
    ):

        samples = [
            sample
            for sample in self.buffer
            if sample.get("game_id")
            == game_id
        ]

        samples.sort(
            key=lambda sample:
            sample.get(
                "step_index",
                0,
            )
        )

        return samples

    # ============================================================
    # LENGTH
    # ============================================================

    def __len__(self):
        return len(self.buffer)

    # ============================================================
    # SAVE
    # ============================================================

    def save(self, path):

        data = {

            "format_version":
                self.FORMAT_VERSION,

            "capacity":
                self.capacity,

            "buffer":
                self.buffer,

            "position":
                self.position,

            "games":
                self.games,

            "game_sample_counts":
                self.game_sample_counts,

            "next_game_id":
                self.next_game_id,

            "metadata":
                self.metadata,
        }

        output_dir = os.path.dirname(
            path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        temp_path = (
            path + ".tmp"
        )

        with open(
            temp_path,
            "wb",
        ) as f:

            pickle.dump(
                data,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        os.replace(
            temp_path,
            path,
        )

    # ============================================================
    # LOAD
    # ============================================================

    @classmethod
    def load(
        cls,
        path,
    ):

        with open(
            path,
            "rb",
        ) as f:

            data = pickle.load(
                f
            )

        format_version = data.get(
            "format_version",
            1,
        )

        # ========================================================
        # V3
        # ========================================================

        if format_version >= 3:

            replay_buffer = cls(
                capacity=data["capacity"],
                metadata=data.get(
                    "metadata",
                    {},
                ),
            )

            replay_buffer.buffer = (
                data["buffer"]
            )

            replay_buffer.position = (
                data["position"]
            )

            replay_buffer.games = (
                data.get(
                    "games",
                    {},
                )
            )

            replay_buffer.game_sample_counts = (
                data.get(
                    "game_sample_counts",
                    {},
                )
            )

            replay_buffer.next_game_id = (
                data.get(
                    "next_game_id",
                    0,
                )
            )

            return replay_buffer

        # ========================================================
        # LEGACY V1 / V2
        # ========================================================

        replay_buffer = cls(
            capacity=data["capacity"]
        )

        old_buffer = data[
            "buffer"
        ]

        # --------------------------------------------------------
        # Convert:
        #
        #     (observation, policy, value)
        #
        # into explicitly named legacy records.
        #
        # We DO NOT pretend the missing information exists.
        # --------------------------------------------------------

        for sample in old_buffer:

            if isinstance(
                sample,
                dict,
            ):

                converted = dict(
                    sample
                )

            else:

                observation, policy, value = (
                    sample
                )

                converted = {

                    "sample_version": 1,

                    "legacy": True,

                    "observation":
                        observation,

                    "policy_target":
                        policy,

                    "value_target_legacy":
                        value,
                }

            replay_buffer.buffer.append(
                converted
            )

        replay_buffer.position = (
            data.get(
                "position",
                len(replay_buffer.buffer)
                % replay_buffer.capacity,
            )
        )

        # --------------------------------------------------------
        # Recover your existing V2 game records where possible.
        # --------------------------------------------------------

        game_records = data.get(
            "game_records",
            [],
        )

        for record in game_records:

            record = dict(record)

            game_id = record[
                "game_id"
            ]

            replay_buffer.games[
                game_id
            ] = record

            start = record[
                "start_index"
            ]

            end = record[
                "end_index"
            ]

            for i in range(
                start,
                min(
                    end,
                    len(
                        replay_buffer.buffer
                    ),
                ),
            ):

                replay_buffer.buffer[i][
                    "game_id"
                ] = game_id

                replay_buffer.buffer[i][
                    "split"
                ] = record.get(
                    "split"
                )

        counts = Counter(
            sample.get("game_id")
            for sample
            in replay_buffer.buffer
            if sample.get("game_id")
            is not None
        )

        replay_buffer.game_sample_counts = (
            dict(counts)
        )

        if replay_buffer.games:

            replay_buffer.next_game_id = (
                max(
                    replay_buffer.games
                )
                + 1
            )

        # Preserve miscellaneous old dataset metadata.
        replay_buffer.metadata = {

            "legacy_format_version":
                format_version,

            "successful_games":
                data.get(
                    "successful_games"
                ),

            "failed_seeds":
                data.get(
                    "failed_seeds",
                    [],
                ),

            "validation_fraction":
                data.get(
                    "validation_fraction"
                ),

            "split_seed":
                data.get(
                    "split_seed"
                ),
        }

        return replay_buffer