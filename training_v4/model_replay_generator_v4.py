import copy
import numpy as np

from splendor_v1.training_v2.replay_generator import (
    ReplayGenerator
)


class ModelReplayGeneratorV4(
    ReplayGenerator
):

    def __init__(
        self,
        env,
        mcts,
        replay_buffer,
        state_serializer,
        temperature=1.0,
        temperature_fn=None,
        add_root_noise=True,
        root_noise_fn=None,

        teacher_mode=False,
        action_space_version=1,
        max_game_steps=300,

    ):

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            state_serializer=state_serializer,
            action_space_version=action_space_version,
        )

        self.mcts = mcts

        self.temperature = float(
            temperature
        )

        self.temperature_fn = (
            temperature_fn
        )

        self.add_root_noise = bool(
            add_root_noise
        )

        self.root_noise_fn = (
            root_noise_fn
        )

        self.teacher_mode = bool(
            teacher_mode
        )

        self.max_game_steps = int(
            max_game_steps
        )

        if self.max_game_steps < 1:

            raise ValueError(
                "max_game_steps must be >= 1."
            )

        self.action_rng = (
            np.random.default_rng()
        )


    def get_add_root_noise(
        self,
        state,
    ):
        """
        Decide whether Dirichlet root noise should
        be enabled for the current Splendor turn.
        """

        if (
            self.root_noise_fn
            is not None
        ):

            return bool(
                self.root_noise_fn(
                    state.turn_number
                )
            )

        return bool(
            self.add_root_noise
    )

    # ============================================================
    # TEMPERATURE
    # ============================================================

    def get_temperature(
        self,
        state,
    ):
        """
        Resolve the self-play temperature for the
        current Splendor position.

        The schedule uses turn_number rather than
        replay step_index so forced sub-decisions
        such as overflow discards and noble claims
        do not advance the temperature schedule.
        """

        if (
            self.temperature_fn
            is not None
        ):

            return float(
                self.temperature_fn(
                    state.turn_number
                )
            )

        return float(
            self.temperature
        )





    # ============================================================
    # ROOT SEARCH DATA
    # ============================================================

    def extract_root_statistics(
        self,
        root,
    ):
        """
        Preserve the raw information produced by MCTS.

        IMPORTANT:

        policy_action_ids
        policy_actions
        visit_counts
        search_priors
        search_value_sums
        search_q_values

        all share the SAME ordering.

        We preserve both the old canonical action ID and
        the semantic action so future action-space layouts
        can migrate old replay data.
        """

        children = list(
            root.children
        )

        action_ids = []
        actions = []

        visit_counts = []

        network_priors = []
        search_priors = []

        value_sums = []
        q_values = []

        for child in children:

            # ----------------------------------------------------
            # ACTION
            # ----------------------------------------------------

            action_id = (
                self.action_to_id(
                    child.action
                )
            )

            serialized_action = (
                self.serialize_action(
                    child.action
                )
            )

            # ----------------------------------------------------
            # SEARCH STATISTICS
            # ----------------------------------------------------

            visits = int(
                getattr(
                    child,
                    "visits",
                    0,
                )
            )

            search_prior = float(
                getattr(
                    child,
                    "prior",
                    0.0,
                )
            )

            network_prior = float(
                getattr(
                    child,
                    "network_prior",
                    search_prior,
                )
            )

            value_sum = float(
                getattr(
                    child,
                    "value",
                    0.0,
                )
            )

            if visits > 0:

                q_value = (
                    value_sum
                    / visits
                )

            else:

                q_value = 0.0

            # ----------------------------------------------------
            # STORE ALIGNED DATA
            # ----------------------------------------------------

            action_ids.append(
                action_id
            )

            actions.append(
                serialized_action
            )

            visit_counts.append(
                visits
            )

            network_priors.append(
                network_prior
            )

            search_priors.append(
                search_prior
            )

            value_sums.append(
                value_sum
            )

            q_values.append(
                q_value
            )

        # ========================================================
        # ROOT STATISTICS
        # ========================================================

        root_visits = int(
            getattr(
                root,
                "visits",
                0,
            )
        )

        root_value_sum = float(
            getattr(
                root,
                "value",
                0.0,
            )
        )

        if root_visits > 0:

            root_value = (
                root_value_sum
                / root_visits
            )

        else:

            root_value = 0.0

        return {

            # ----------------------------------------------------
            # ACTIONS
            # ----------------------------------------------------

            "policy_action_ids":
                np.asarray(
                    action_ids,
                    dtype=np.int32,
                ),

            "policy_actions":
                actions,

            # ----------------------------------------------------
            # SEARCH POLICY
            # ----------------------------------------------------

            "visit_counts":
                np.asarray(
                    visit_counts,
                    dtype=np.int32,
                ),

            "network_priors":
                np.asarray(
                    network_priors,
                    dtype=np.float32,
                ),

            "search_priors":
                np.asarray(
                    search_priors,
                    dtype=np.float32,
                ),
            # ----------------------------------------------------
            # SEARCH VALUES
            # ----------------------------------------------------

            "search_value_sums":
                np.asarray(
                    value_sums,
                    dtype=np.float32,
                ),

            "search_q_values":
                np.asarray(
                    q_values,
                    dtype=np.float32,
                ),

            # ----------------------------------------------------
            # ROOT
            # ----------------------------------------------------

            "root_visits":
                root_visits,

            "root_value_sum":
                root_value_sum,

            "root_value":
                float(
                    root_value
                ),

            "search_child_count":
                len(children),
        }

    # ============================================================
    # VALIDATE SEARCH DATA
    # ============================================================

    def validate_search_data(
        self,
        sample,
    ):
        """
        Verify that MCTS search output is compatible with
        the position that was captured before search.

        Fail immediately rather than allowing corrupted
        replay samples into the buffer.
        """

        legal_id_list = [
            int(action_id)
            for action_id
            in sample[
                "legal_action_ids"
            ]
        ]

        policy_id_list = [
            int(action_id)
            for action_id
            in sample[
                "policy_action_ids"
            ]
        ]

        legal_ids = set(
            legal_id_list
        )

        policy_ids = set(
            policy_id_list
        )

        if len(legal_ids) != len(
            legal_id_list
        ):

            raise RuntimeError(
                "legal_action_ids contains duplicate "
                "canonical action IDs."
            )

        if len(policy_ids) != len(
            policy_id_list
        ):

            raise RuntimeError(
                "policy_action_ids contains duplicate "
                "canonical action IDs."
            )

        # Model 4 training relies on the candidate IDs, visit counts,
        # priors, and value arrays sharing exactly the same ordering.
        if policy_id_list != legal_id_list:

            missing_ids = (
                legal_ids
                - policy_ids
            )

            extra_ids = (
                policy_ids
                - legal_ids
            )

            raise RuntimeError(
                "MCTS root action ordering does not exactly match "
                "legal_action_ids. "
                f"Missing IDs: {sorted(missing_ids)} | "
                f"Extra IDs: {sorted(extra_ids)} | "
                f"legal_order={legal_id_list} | "
                f"policy_order={policy_id_list}"
            )
        
        # All parallel MCTS arrays must line up.
        expected_length = len(
            sample[
                "policy_action_ids"
            ]
        )

        aligned_fields = [
            "policy_actions",
            "visit_counts",
            "network_priors",
            "search_priors",
            "search_value_sums",
            "search_q_values",
        ]

        for field in aligned_fields:

            if len(
                sample[field]
            ) != expected_length:

                raise RuntimeError(
                    f"MCTS replay field '{field}' "
                    "does not align with "
                    "policy_action_ids."
                )

    # ============================================================
    # ACTION SELECTION
    # ============================================================

    def select_action_from_root(
        self,
        root,
        temperature,
        fallback_action=None,
    ):
        """
        Select an environment action from root visits.

        temperature <= 0:
            choose max visits

        temperature > 0:
            sample from visits^(1 / temperature)
        """

        children = list(
            root.children
        )

        if not children:

            if fallback_action is not None:

                return (
                    fallback_action
                )

            raise RuntimeError(
                "MCTS root has no children."
            )

        visits = np.asarray(
            [
                child.visits
                for child
                in children
            ],
            dtype=np.float64,
        )

        # ========================================================
        # NO SEARCH VISITS
        # ========================================================

        if visits.sum() <= 0:

            if fallback_action is not None:

                return (
                    fallback_action
                )

            return (
                children[0].action
            )

        # ========================================================
        # GREEDY
        # ========================================================

        if temperature <= 1e-8:

            index = int(
                np.argmax(
                    visits
                )
            )

            return (
                children[index].action
            )

        # ========================================================
        # TEMPERATURE SAMPLING
        # ========================================================

        exponent = (
            1.0
            / temperature
        )

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):

            weights = np.power(
                visits,
                exponent,
            )

        total = (
            weights.sum()
        )

        # Numerical fallback.
        if (
            not np.isfinite(total)
            or total <= 0
        ):

            index = int(
                np.argmax(
                    visits
                )
            )

            return (
                children[index].action
            )

        probabilities = (
            weights
            / total
        )

        index = int(
            self.action_rng.choice(
                len(children),
                p=probabilities,
            )
        )

        return (
            children[index].action
        )

    # ============================================================
    # ROOT REUSE
    # ============================================================

    @staticmethod
    def find_selected_child(
        root,
        action,
    ):

        for child in root.children:

            if child.action == action:

                return child

        return None

    # ============================================================
    # SEARCH CONFIG METADATA
    # ============================================================

    def get_search_metadata(
        self,
    ):

        metadata = {

            "add_root_noise":
                self.add_root_noise,

            "teacher_mode":
                self.teacher_mode,

            "replay_generator_version":
                4,

            "max_game_steps":
                self.max_game_steps,
        }


        if (
            self.root_noise_fn
            is None
        ):

            metadata[
                "root_noise"
            ] = bool(
                self.add_root_noise
            )

        else:

            metadata[
                "root_noise_schedule"
            ] = getattr(
                self.root_noise_fn,
                "__name__",
                "custom",
            )



        # --------------------------------------------------------
        # Preserve known MCTS configuration fields.
        #
        # Different versions of MCTS may expose different names,
        # so only save fields that actually exist.
        # --------------------------------------------------------

        fields = [
            "simulations",
            "c_puct",
            "dirichlet_alpha",
            "dirichlet_epsilon",
            "rollout_type",
            "selection_type",
        ]

        for field in fields:

            if hasattr(
                self.mcts,
                field,
            ):

                value = getattr(
                    self.mcts,
                    field,
                )

                # Keep metadata pickle-safe/simple.
                if isinstance(
                    value,
                    np.generic,
                ):

                    value = (
                        value.item()
                    )

                metadata[
                    field
                ] = value

        # --------------------------------------------------------
        # TEMPERATURE
        # --------------------------------------------------------

        if (
            self.temperature_fn
            is None
        ):

            metadata[
                "temperature"
            ] = float(
                self.temperature
            )

        else:

            metadata[
                "temperature_schedule"
            ] = getattr(
                self.temperature_fn,
                "__name__",
                "custom",
            )

        return metadata

    # ============================================================
    # GENERATE GAME
    # ============================================================

    def generate_game(
        self,
        seed=None,
        split=None,
        model_generation=4,
        model_checkpoint=None,
        extra_game_metadata=None,
    ):

        # ========================================================
        # PER-GAME RANDOM NUMBER GENERATORS
        # ========================================================

        if seed is None:

            # Non-reproducible run.
            self.action_rng = (
                np.random.default_rng()
            )

            self.mcts.rng = (
                np.random.default_rng()
            )

        else:

            # Derive independent deterministic RNG streams
            # from the game's seed.
            seed_sequence = (
                np.random.SeedSequence(
                    int(seed)
                )
            )

            (
                mcts_seed,
                action_seed,
            ) = seed_sequence.spawn(2)

            # Dirichlet noise.
            self.mcts.rng = (
                np.random.default_rng(
                    mcts_seed
                )
            )

            # Temperature action selection.
            self.action_rng = (
                np.random.default_rng(
                    action_seed
                )
            )




        # ========================================================
        # RESET
        # ========================================================

        state = self.reset(
            seed=seed
        )

        trajectory = []

        terminated = False

        step_index = 0

        # Used for MCTS tree reuse.
        current_root = None

        # ========================================================
        # PLAY GAME
        # ========================================================

        while not terminated:

            # ====================================================
            # SAFETY: PATHOLOGICALLY LONG GAME
            # ====================================================
            #
            # A single non-terminating seed must never be able to
            # block an entire long self-play run. Because step_index
            # counts MAIN decisions AND forced sub-decisions, it is
            # the correct watchdog for the replay generator.
            # ====================================================

            if step_index >= self.max_game_steps:

                scores = [
                    getattr(
                        player,
                        "points",
                        None,
                    )
                    for player
                    in state.players
                ]

                raise RuntimeError(
                    "ModelReplayGeneratorV4 exceeded maximum "
                    "game length. "
                    f"seed={seed}, "
                    f"step_index={step_index}, "
                    f"turn_number={state.turn_number}, "
                    f"current_player={state.current_player}, "
                    f"node_type={state.node_type}, "
                    f"scores={scores}, "
                    f"max_game_steps={self.max_game_steps}"
                )

            # ----------------------------------------------------
            # Capture state BEFORE search and BEFORE action.
            # ----------------------------------------------------

            sample = (
                self.build_base_sample(
                    state=state,
                    step_index=step_index,
                )
            )

            # ====================================================
            # MCTS SEARCH
            # ====================================================

            use_root_noise = (
                self.get_add_root_noise(
                    state
                )
            )

            sample[
                "root_noise_enabled"
            ] = bool(
                use_root_noise
            )


            search_result = (
                self.mcts.search(
                    self.env,
                    state,
                    root=current_root,
                    return_root=True,
                    add_root_noise=(
                        use_root_noise
                    ),
                    teacher_mode=(
                        self.teacher_mode
                    ),
                )
            )

            if (
                not isinstance(
                    search_result,
                    tuple,
                )
                or len(
                    search_result
                ) != 2
            ):

                raise RuntimeError(
                    "Expected MCTS.search("
                    "return_root=True) to return "
                    "(action, root)."
                )

            (
                search_action,
                root,
            ) = search_result

            if root is None:

                raise RuntimeError(
                    "MCTS returned root=None."
                )

            # ====================================================
            # SAVE RAW SEARCH INFORMATION
            # ====================================================

            search_data = (
                self.extract_root_statistics(
                    root
                )
            )

            sample.update(
                search_data
            )

            sample[
                "policy_source"
            ] = "mcts"

            # ----------------------------------------------------
            # Validate MCTS children against the state captured
            # before the search.
            # ----------------------------------------------------

            self.validate_search_data(
                sample
            )

            # ====================================================
            # CHOOSE SELF-PLAY ACTION
            # ====================================================

            temperature = (
                self.get_temperature(
                    state
                )
            )

            action = (
                self.select_action_from_root(
                    root=root,
                    temperature=temperature,
                    fallback_action=(
                        search_action
                    ),
                )
            )

            if action is None:

                raise RuntimeError(
                    "Model/MCTS returned None "
                    "before termination."
                )

            # ----------------------------------------------------
            # Let ReplayGenerator record BOTH:
            #
            #     canonical ID
            #     semantic action
            # ----------------------------------------------------

            self.record_chosen_action(
                sample,
                action,
            )

            sample[
                "temperature"
            ] = float(
                temperature
            )

            # ====================================================
            # VERIFY CHOSEN ACTION IS LEGAL
            # ====================================================

            legal_ids = set(
                int(action_id)
                for action_id
                in sample[
                    "legal_action_ids"
                ]
            )

            chosen_action_id = int(
                sample[
                    "chosen_action_id"
                ]
            )

            if (
                chosen_action_id
                not in legal_ids
            ):

                raise RuntimeError(
                    "Selected self-play action "
                    "was not legal in the captured "
                    f"position. action_id="
                    f"{chosen_action_id}"
                )

            # ====================================================
            # FIND CHILD BEFORE STEP
            # ====================================================

            selected_child = (
                self.find_selected_child(
                    root,
                    action,
                )
            )

            previous_player = (
                state.current_player
            )


            # ====================================================
            # STEP ENVIRONMENT
            # ====================================================

            (
                reward,
                terminated,
                _,
            ) = self.step(
                action
            )

            # ----------------------------------------------------
            # Let ReplayGenerator record transition outcome.
            # ----------------------------------------------------

            self.record_transition(
                sample,
                reward,
                terminated,
            )

            # Position is now complete.
            trajectory.append(
                sample
            )

            # ====================================================
            # NEXT STATE
            # ====================================================

            state = self.env.state


            # ====================================================
            # TREE REUSE
            # ====================================================

            if (
                not terminated
                and selected_child
                is not None
            ):

                # ------------------------------------------------
                # MCTS stores values from the perspective of the
                # player who was root_player during the previous
                # search.
                #
                # If the player-to-move has changed, the entire
                # retained subtree must be converted to the new
                # root player's perspective.
                #
                # This is correct for our current 2-player,
                # zero-sum Splendor setup.
                # ------------------------------------------------

                if (
                    state.current_player
                    != previous_player
                ):

                    self.mcts.flip_tree_values(
                        selected_child
                    )

                selected_child.parent = None

                current_root = (
                    selected_child
                )

            else:

                current_root = None


            step_index += 1

        # ========================================================
        # GAME COMPLETE
        # ========================================================

        metadata = {

            "model_generation":
                model_generation,

            "model_checkpoint":
                model_checkpoint,

            "search":
                self.get_search_metadata(),
        }

        if extra_game_metadata:

            metadata.update(
                copy.deepcopy(
                    extra_game_metadata
                )
            )

        game_metadata = (
            self.build_game_metadata(
                state=state,
                seed=seed,
                source="model_self_play",
                num_positions=len(
                    trajectory
                ),
                split=split,
                extra_metadata=metadata,
            )
        )

        # ========================================================
        # FINAL SANITY CHECKS
        # ========================================================

        if len(
            trajectory
        ) == 0:

            raise RuntimeError(
                "Self-play completed with "
                "an empty trajectory."
            )

        if not trajectory[-1][
            "terminated_after_action"
        ]:

            raise RuntimeError(
                "Game completed but the final "
                "replay sample is not marked "
                "terminated."
            )

        winner_ids = (
            game_metadata.get(
                "winner_ids",
                [],
            )
        )

        if not winner_ids:

            raise RuntimeError(
                "Completed game has no winner IDs."
            )

        # ========================================================
        # COMMIT ONLY AFTER SUCCESSFUL COMPLETION
        # ========================================================

        return self.commit_game(
            trajectory=trajectory,
            game_metadata=game_metadata,
        )

# ============================================================
# COMPATIBILITY ALIAS
# ============================================================
#
# Existing training code often imports:
#
#     from ...model_replay_generator_v4 import ModelReplayGenerator
#
# Keep that import ergonomic while still exposing the generation-
# specific class name.
# ============================================================

ModelReplayGenerator = ModelReplayGeneratorV4
