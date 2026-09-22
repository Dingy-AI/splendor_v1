from abc import ABC, abstractmethod
import copy

import numpy as np

from splendor_v1.env.core.action_constants import (
    ACTION_SPACE_SIZE
)

from splendor_v1.training_v2.action_serializer import (
    serialize_action,
)


class ReplayGenerator(ABC):
    """
    Base class for replay-data generators.

    Responsibilities:
        - reset / step the environment
        - encode observations
        - serialize raw game states
        - capture legal actions
        - capture canonical action IDs
        - capture semantic action representations
        - capture player / node information
        - extract final winners / scores
        - commit COMPLETE games to ReplayBuffer

    Subclasses decide:
        - how actions are chosen
        - how policy/search information is produced
    """

    def __init__(
        self,
        env,
        replay_buffer,
        state_serializer,
        action_space_version=1,
    ):

        self.env = env

        self.replay_buffer = (
            replay_buffer
        )

        # ----------------------------------------------------
        # STATE SERIALIZER
        # ----------------------------------------------------

        if state_serializer is None:

            raise ValueError(
                "ReplayGenerator requires a "
                "state_serializer."
            )

        self.state_serializer = (
            state_serializer
        )

        # ----------------------------------------------------
        # ACTION-SPACE METADATA
        # ----------------------------------------------------

        self.action_space_version = int(
            action_space_version
        )

        self.action_space_size = int(
            ACTION_SPACE_SIZE
        )

    # ============================================================
    # ENVIRONMENT HELPERS
    # ============================================================

    def reset(
        self,
        seed=None,
    ):

        if seed is None:

            result = self.env.reset()

        else:

            result = self.env.reset(
                seed=seed
            )

        # --------------------------------------------------------
        # Support either:
        #
        #     state = env.reset()
        #
        # or Gym-style reset APIs.
        # --------------------------------------------------------

        if hasattr(
            result,
            "current_player",
        ):

            return result

        return self.env.state

    # ============================================================
    # OBSERVATION
    # ============================================================

    def encode_observation(
        self,
        state,
    ):

        observation = (
            self.env
            .observation_encoder
            .encoder(
                state
            )
        )

        return np.asarray(
            observation,
            dtype=np.float32,
        ).copy()

    # ============================================================
    # STATE SERIALIZATION
    # ============================================================

    def serialize_state(
        self,
        state,
    ):

        serialized = (
            self.state_serializer(
                state
            )
        )

        return copy.deepcopy(
            serialized
        )

    # ============================================================
    # ACTION SERIALIZATION
    # ============================================================

    def serialize_action(
        self,
        action,
    ):

        return copy.deepcopy(
            serialize_action(
                action
            )
        )

    # ============================================================
    # CANONICAL ACTION ID
    # ============================================================

    def action_to_id(
        self,
        action,
    ):

        action_id = (
            self.env.action_to_id(
                action
            )
        )

        return int(
            action_id
        )

    # ============================================================
    # LEGAL ACTIONS
    # ============================================================

    def get_legal_actions(
        self,
        state,
    ):

        legal_actions = (
            self.env._legal_actions(
                state
            )
        )

        return list(
            legal_actions
        )

    # ============================================================
    # LEGAL ACTION IDS
    # ============================================================

    def get_action_ids(
        self,
        actions,
    ):

        return np.asarray(
            [
                self.action_to_id(
                    action
                )
                for action
                in actions
            ],
            dtype=np.int32,
        )

    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        action,
    ):
        """
        Returns:

            reward
            terminated
            info
        """

        result = self.env.step(
            action
        )

        reward = 0.0

        terminated = False

        info = {}

        # --------------------------------------------------------
        # GYM / GYMNASIUM STYLE
        # --------------------------------------------------------

        if isinstance(
            result,
            tuple,
        ):

            # ----------------------------------------------------
            # Gymnasium:
            #
            # obs
            # reward
            # terminated
            # truncated
            # info
            # ----------------------------------------------------

            if len(result) == 5:

                (
                    _,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = result

                terminated = (
                    terminated
                    or truncated
                )

            # ----------------------------------------------------
            # Older Gym:
            #
            # obs
            # reward
            # done
            # info
            # ----------------------------------------------------

            elif len(result) == 4:

                (
                    _,
                    reward,
                    terminated,
                    info,
                ) = result

            else:

                raise RuntimeError(
                    "Unexpected env.step() "
                    f"tuple length: "
                    f"{len(result)}"
                )

        # --------------------------------------------------------
        # NON-GYM STYLE
        # --------------------------------------------------------

        else:

            # GameState already exposes:
            #
            #     game_over
            #
            # so use that as the authoritative
            # termination flag.

            terminated = bool(
                getattr(
                    self.env.state,
                    "game_over",
                    False,
                )
            )

        return (
            self._copy_value(
                reward
            ),
            bool(
                terminated
            ),
            copy.deepcopy(
                info
            ),
        )

    # ============================================================
    # POSITION SAMPLE
    # ============================================================

    def build_base_sample(
        self,
        state,
        step_index,
    ):
        """
        Capture everything that is independent
        of the agent / search method.

        IMPORTANT:

        Legal actions are generated ONCE here.

        We then preserve both:

            canonical action IDs
            semantic actions

        This avoids calling _legal_actions twice
        and guarantees the two representations
        correspond exactly.
        """

        # --------------------------------------------------------
        # NODE TYPE
        # --------------------------------------------------------

        node_type = getattr(
            state,
            "node_type",
            None,
        )

        if hasattr(
            node_type,
            "name",
        ):

            node_type = (
                node_type.name
            )

        # --------------------------------------------------------
        # LEGAL ACTIONS
        # --------------------------------------------------------

        legal_actions = (
            self.get_legal_actions(
                state
            )
        )

        if not legal_actions:

            raise RuntimeError(
                "No legal actions available "
                "for a non-terminal replay "
                "position."
            )

        legal_action_ids = (
            self.get_action_ids(
                legal_actions
            )
        )

        serialized_legal_actions = [
            self.serialize_action(
                action
            )
            for action
            in legal_actions
        ]

        # --------------------------------------------------------
        # SAMPLE
        # --------------------------------------------------------

        sample = {

            "sample_version": 1,

            # ====================================================
            # CURRENT MODEL REPRESENTATION
            # ====================================================

            "observation":
                self.encode_observation(
                    state
                ),

            # ====================================================
            # RAW SEMANTIC STATE
            # ====================================================

            "state":
                self.serialize_state(
                    state
                ),

            # ====================================================
            # POSITION IDENTITY
            # ====================================================

            "current_player":
                int(
                    state.current_player
                ),

            "node_type":
                node_type,

            "turn_number":
                int(
                    state.turn_number
                ),

            "step_index":
                int(
                    step_index
                ),

            # ====================================================
            # LEGAL ACTIONS
            #
            # Keep BOTH representations.
            # ====================================================

            "legal_action_ids":
                legal_action_ids,

            "legal_actions":
                serialized_legal_actions,

            # ====================================================
            # CHOSEN ACTION
            #
            # Filled after the agent/search decides.
            # ====================================================

            "chosen_action_id":
                None,

            "chosen_action":
                None,

            # ====================================================
            # TRANSITION
            # ====================================================

            "reward":
                None,

            "terminated_after_action":
                False,
        }

        return sample

    # ============================================================
    # RECORD CHOSEN ACTION
    # ============================================================

    def record_chosen_action(
        self,
        sample,
        action,
    ):
        """
        Attach both forms of the chosen action.

        Example:

            chosen_action_id = 527

        AND

            chosen_action = {
                "action_type": "BUY_VISIBLE",
                ...
            }

        This allows future action-space layouts
        to recover the original semantic action.
        """

        if action is None:

            raise ValueError(
                "Cannot record action=None."
            )

        sample[
            "chosen_action_id"
        ] = self.action_to_id(
            action
        )

        sample[
            "chosen_action"
        ] = self.serialize_action(
            action
        )

    # ============================================================
    # RECORD TRANSITION RESULT
    # ============================================================

    def record_transition(
        self,
        sample,
        reward,
        terminated,
    ):

        sample[
            "reward"
        ] = self._copy_value(
            reward
        )

        sample[
            "terminated_after_action"
        ] = bool(
            terminated
        )

    # ============================================================
    # TERMINAL INFORMATION
    # ============================================================

    def get_winner_ids(
        self,
        state,
    ):

        winners = getattr(
            state,
            "winners",
            [],
        )

        if winners is None:

            return []

        return [
            int(
                player_id
            )
            for player_id
            in winners
        ]

    # ============================================================
    # FINAL SCORES
    # ============================================================

    def get_final_scores(
        self,
        state,
    ):

        players = getattr(
            state,
            "players",
            None,
        )

        if players is None:

            return None

        scores = []

        for player in players:

            points = getattr(
                player,
                "points",
                None,
            )

            if points is None:

                scores.append(
                    None
                )

            else:

                scores.append(
                    int(
                        points
                    )
                )

        return scores

    # ============================================================
    # GAME METADATA
    # ============================================================

    def build_game_metadata(
        self,
        state,
        seed,
        source,
        num_positions,
        split=None,
        extra_metadata=None,
    ):

        metadata = {

            # ----------------------------------------------------
            # GAME IDENTITY
            # ----------------------------------------------------

            "seed":
                seed,

            "source":
                source,

            "split":
                split,

            # ----------------------------------------------------
            # RESULT
            # ----------------------------------------------------

            "winner_ids":
                self.get_winner_ids(
                    state
                ),

            "final_scores":
                self.get_final_scores(
                    state
                ),

            "num_positions":
                int(
                    num_positions
                ),

            "final_turn_number":
                int(
                    state.turn_number
                ),

            # ----------------------------------------------------
            # ACTION-SPACE INTERPRETATION
            # ----------------------------------------------------

            "action_space_version":
                self.action_space_version,

            "action_space_size":
                self.action_space_size,
        }

        # --------------------------------------------------------
        # OPTIONAL GENERATOR-SPECIFIC METADATA
        # --------------------------------------------------------

        if extra_metadata:

            metadata.update(
                copy.deepcopy(
                    extra_metadata
                )
            )

        return metadata

    # ============================================================
    # COMMIT COMPLETE GAME
    # ============================================================

    def commit_game(
        self,
        trajectory,
        game_metadata,
    ):
        """
        Only completed games should reach here.

        Failed games therefore never partially
        pollute the replay buffer.
        """

        if len(
            trajectory
        ) == 0:

            raise RuntimeError(
                "Cannot commit an empty "
                "trajectory."
            )

        game_id = (
            self.replay_buffer.add_game(
                samples=trajectory,
                game_metadata=game_metadata,
            )
        )

        return {

            "game_id":
                game_id,

            "num_positions":
                len(
                    trajectory
                ),

            "game_metadata":
                game_metadata,
        }

    # ============================================================
    # SMALL HELPERS
    # ============================================================

    @staticmethod
    def _copy_value(
        value,
    ):

        if isinstance(
            value,
            np.ndarray,
        ):

            return value.copy()

        if isinstance(
            value,
            np.generic,
        ):

            return value.item()

        return copy.deepcopy(
            value
        )

    # ============================================================
    # SUBCLASS API
    # ============================================================

    @abstractmethod
    def generate_game(
        self,
        seed=None,
        **kwargs,
    ):

        raise NotImplementedError