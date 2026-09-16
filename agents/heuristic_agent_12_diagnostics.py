from collections import Counter, defaultdict

from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12
from splendor_v1.env.core.enums import NodeType
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent12Diagnostics(HeuristicAgent12):
    """
    Drop-in diagnostic version of HeuristicAgent12.

    IMPORTANT:
        This class does not change H12's candidate generation,
        strategy classification, rollout calculation, or final
        action ranking.

        It calls the exact same inherited H12 logic and only records
        information about the action H12 already selected.

    Tracks
    ------
    Strategic MAIN_DECISION nodes:
        - noble / reserve / neutral strategy counts
        - how often H12 chose an original H4 candidate
        - how often H12 chose a strategic extra
        - which strategy produced the selected extra
        - selected extra action type
        - how many strategic extras were proposed
        - average noble / reserve classifier scores
        - strategy/source cross-table
        - stats by player index

    Forced nodes are counted separately and are excluded from the
    strategic percentages.

    Typical usage
    -------------
        agent12 = HeuristicAgent12Diagnostics(
            num_rollouts=8,
            num_strategy_moves=2,
        )

        # Run your normal evaluation loop.

        print(agent12.format_diagnostic_summary())

    Or:

        stats = agent12.get_diagnostic_stats()
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.reset_diagnostic_stats()

    # ============================================================
    # PUBLIC ACTION SELECTION
    # ============================================================

    def select_action(
        self,
        env,
        state,
    ):
        """
        Same decision rule as H12.

        We intentionally reproduce H12.select_action rather than
        calling super().select_action() because we need access to
        the already-computed evaluated candidate list before the
        method returns.

        The max key is identical to H12:
            1. rollout value
            2. cheap score tie-break
        """

        evaluated = (
            self._get_rollout_evaluated_candidates(
                env,
                state,
            )
        )

        if not evaluated:
            return None

        action, rollout_value, cheap_score = max(
            evaluated,
            key=lambda item: (
                item[1],
                item[2],
            ),
        )

        self._record_decision(
            state=state,
            selected_action=action,
            selected_rollout_value=rollout_value,
            selected_cheap_score=cheap_score,
        )

        self._update_overflow_diagnostics(
            state,
            action,
        )

        return action

    # ============================================================
    # RESET
    # ============================================================

    def reset_diagnostic_stats(
        self,
    ):
        self._diag = {
            "total_select_action_calls":
                0,

            "forced_node_calls":
                0,

            "forced_node_type_counts":
                Counter(),

            "overflow_events":
                0,

            "overflow_preceding_action_counts":
                Counter(),

            "overflow_start_token_counts":
                Counter(),

            "overflow_start_token_sum":
                0,

            "overflow_start_token_observations":
                0,

            "overflow_size_counts":
                Counter(),

            "overflow_size_sum":
                0,

            "overflow_gained_color_counts":
                Counter(),

            "overflow_discarded_color_counts":
                Counter(),

            "overflow_events_rediscarding_gained_color":
                0,

            "overflow_rediscard_token_count":
                0,

            "overflow_rediscard_color_counts":
                Counter(),

            "strategic_decisions":
                0,

            "strategy_counts":
                Counter(),

            "selected_source_counts":
                Counter(),

            "selected_extra_strategy_counts":
                Counter(),

            "selected_extra_action_type_counts":
                Counter(),

            "strategy_source_counts":
                Counter(),

            "extras_proposed_total":
                0,

            "decisions_with_any_extra":
                0,

            "noble_score_sum":
                0.0,

            "reserve_score_sum":
                0.0,

            "strategy_score_observations":
                0,

            "selected_rollout_value_sum":
                0.0,

            "selected_rollout_value_observations":
                0,

            "by_player":
                defaultdict(
                    lambda: {
                        "strategic_decisions":
                            0,

                        "strategy_counts":
                            Counter(),

                        "selected_source_counts":
                            Counter(),

                        "selected_extra_strategy_counts":
                            Counter(),
                    }
                ),
        }

        self._pending_resource_action = None
        self._active_overflow_episode = None


    # ============================================================
    # OVERFLOW / RESOURCE-ACTION DIAGNOSTICS
    # ============================================================

    def _diag_color_name(self, color):
        return getattr(color, "name", str(color))

    def _diag_token_count(self, state):
        player = state.players[state.current_player]
        return int(sum(player.gems.values()))

    def _diag_action_color_names(self, action):
        for attr in ("gem_colors", "discard_colors", "colors"):
            colors = getattr(action, attr, None)
            if colors is not None:
                try:
                    return [
                        self._diag_color_name(color)
                        for color in colors
                    ]
                except TypeError:
                    pass
        return []

    def _diag_gold_available(self, state):
        for color, count in state.bank.items():
            if self._diag_color_name(color) == "GOLD":
                return count > 0
        return False

    def _capture_resource_action_context(self, state, action):
        # A new MAIN_DECISION invalidates stale pending context.
        self._pending_resource_action = None

        action_type = getattr(action, "action_type", None)

        if action_type not in (
            ActionType.TAKE_GEMS,
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_TOP_DECK,
        ):
            return

        if action_type == ActionType.TAKE_GEMS:
            gained_colors = self._diag_action_color_names(action)
        elif self._diag_gold_available(state):
            gained_colors = ["GOLD"]
        else:
            gained_colors = []

        self._pending_resource_action = {
            "action_type": getattr(
                action_type,
                "name",
                str(action_type),
            ),
            "starting_token_count": self._diag_token_count(state),
            "gained_colors": gained_colors,
        }

    def _start_overflow_episode(
        self,
        state,
    ):
        """
        Convert the pending resource-action context into a persistent
        overflow episode.

        The environment discards exactly one gem per OVERFLOW_DISCARD
        node, so one resource action may generate several forced
        discard nodes. Keep the same episode alive until the player is
        back at or below max_gems.
        """

        context = self._pending_resource_action or {}

        current_tokens = self._diag_token_count(state)
        max_gems = int(getattr(state, "max_gems", 10))

        initial_overflow = max(
            0,
            current_tokens - max_gems,
        )

        self._active_overflow_episode = {
            "action_type": context.get(
                "action_type",
                "UNKNOWN",
            ),
            "starting_token_count": context.get(
                "starting_token_count",
                None,
            ),
            "gained_colors": list(
                context.get(
                    "gained_colors",
                    [],
                )
            ),
            "peak_token_count": current_tokens,
            "initial_overflow": initial_overflow,
            "discarded_colors": [],
        }

        # Pending context is now owned by the active episode.
        self._pending_resource_action = None

    def _finish_overflow_episode(self):
        episode = self._active_overflow_episode

        if not episode:
            return

        self._diag[
            "overflow_events"
        ] += 1

        preceding_action = episode.get(
            "action_type",
            "UNKNOWN",
        )

        self._diag[
            "overflow_preceding_action_counts"
        ][preceding_action] += 1

        starting_tokens = episode.get(
            "starting_token_count",
            None,
        )

        if starting_tokens is not None:
            self._diag[
                "overflow_start_token_counts"
            ][starting_tokens] += 1

            self._diag[
                "overflow_start_token_sum"
            ] += starting_tokens

            self._diag[
                "overflow_start_token_observations"
            ] += 1

        initial_overflow = int(
            episode.get(
                "initial_overflow",
                0,
            )
        )

        self._diag[
            "overflow_size_counts"
        ][initial_overflow] += 1

        self._diag[
            "overflow_size_sum"
        ] += initial_overflow

        gained_colors = list(
            episode.get(
                "gained_colors",
                [],
            )
        )

        discarded_colors = list(
            episode.get(
                "discarded_colors",
                [],
            )
        )

        for color_name in gained_colors:
            self._diag[
                "overflow_gained_color_counts"
            ][color_name] += 1

        for color_name in discarded_colors:
            self._diag[
                "overflow_discarded_color_counts"
            ][color_name] += 1

        gained_counter = Counter(
            gained_colors
        )

        discarded_counter = Counter(
            discarded_colors
        )

        rediscard_counter = (
            gained_counter
            & discarded_counter
        )

        rediscard_tokens = sum(
            rediscard_counter.values()
        )

        if rediscard_tokens > 0:
            self._diag[
                "overflow_events_rediscarding_gained_color"
            ] += 1

            self._diag[
                "overflow_rediscard_token_count"
            ] += rediscard_tokens

            self._diag[
                "overflow_rediscard_color_counts"
            ].update(
                rediscard_counter
            )

        self._active_overflow_episode = None

    def _record_overflow_discard_step(
        self,
        state,
        discard_action,
    ):
        if self._active_overflow_episode is None:
            self._start_overflow_episode(
                state
            )

        episode = self._active_overflow_episode

        discarded_colors = (
            self._diag_action_color_names(
                discard_action
            )
        )

        episode[
            "discarded_colors"
        ].extend(
            discarded_colors
        )

        # The action has not been applied yet. The state still includes
        # the gem that is about to be discarded. Since the environment
        # discards one gem at a time, this episode ends when the current
        # state is only one token above max_gems.
        current_tokens = (
            self._diag_token_count(
                state
            )
        )

        max_gems = int(
            getattr(
                state,
                "max_gems",
                10,
            )
        )

        if current_tokens - 1 <= max_gems:
            self._finish_overflow_episode()

    def _update_overflow_diagnostics(
        self,
        state,
        action,
    ):
        node_name = getattr(
            state.node_type,
            "name",
            str(state.node_type),
        )

        if (
            state.node_type
            == NodeType.MAIN_DECISION
        ):
            # If an episode somehow survived into a new strategic
            # decision, finalize it defensively.
            if self._active_overflow_episode is not None:
                self._finish_overflow_episode()

            self._capture_resource_action_context(
                state,
                action,
            )

        elif node_name == "OVERFLOW_DISCARD":
            self._record_overflow_discard_step(
                state,
                action,
            )

        elif node_name == "NOBLE_CLAIM":
            # Noble claim can follow overflow resolution. Do NOT wipe
            # pending/episode state here prematurely.
            if self._active_overflow_episode is not None:
                self._finish_overflow_episode()

        else:
            # Unknown forced nodes: keep active episode if one exists,
            # otherwise clear stale pending context.
            if self._active_overflow_episode is None:
                self._pending_resource_action = None

    # ============================================================
    # RECORD ONE DECISION
    # ============================================================

    def _record_decision(
        self,
        state,
        selected_action,
        selected_rollout_value,
        selected_cheap_score,
    ):
        self._diag[
            "total_select_action_calls"
        ] += 1

        # Forced discard / noble-claim nodes are not the decisions
        # we are trying to diagnose.
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self._diag[
                "forced_node_calls"
            ] += 1

            node_type_name = getattr(
                state.node_type,
                "name",
                str(
                    state.node_type
                ),
            )

            self._diag[
                "forced_node_type_counts"
            ][node_type_name] += 1

            return

        self._diag[
            "strategic_decisions"
        ] += 1

        strategy_debug = (
            self.last_strategy_debug
            or {}
        )

        candidate_debug = (
            self.last_candidate_debug
            or {}
        )

        strategy = strategy_debug.get(
            "strategy",
            self.NEUTRAL,
        )

        self._diag[
            "strategy_counts"
        ][strategy] += 1

        # --------------------------------------------------------
        # Classifier scores
        # --------------------------------------------------------

        noble_score = strategy_debug.get(
            "noble_score",
            None,
        )

        reserve_score = strategy_debug.get(
            "reserve_score",
            None,
        )

        if (
            noble_score is not None
            and reserve_score is not None
        ):
            self._diag[
                "noble_score_sum"
            ] += float(
                noble_score
            )

            self._diag[
                "reserve_score_sum"
            ] += float(
                reserve_score
            )

            self._diag[
                "strategy_score_observations"
            ] += 1

        self._diag[
            "selected_rollout_value_sum"
        ] += float(
            selected_rollout_value
        )

        self._diag[
            "selected_rollout_value_observations"
        ] += 1

        # --------------------------------------------------------
        # Proposed extras
        # --------------------------------------------------------

        extras = candidate_debug.get(
            "strategy_extras",
            [],
        )

        extra_count = len(
            extras
        )

        self._diag[
            "extras_proposed_total"
        ] += extra_count

        if extra_count > 0:
            self._diag[
                "decisions_with_any_extra"
            ] += 1

        # --------------------------------------------------------
        # Determine whether selected action came from original H4
        # candidate set or from H12's strategic additions.
        # --------------------------------------------------------

        base_candidates = (
            candidate_debug.get(
                "base_candidates",
                [],
            )
        )

        selected_id = (
            self._action_identity(
                selected_action
            )
        )

        base_ids = {
            self._action_identity(
                action
            )
            for action, _ in base_candidates
        }

        extra_ids = {
            self._action_identity(
                action
            )
            for action, _ in extras
        }

        if selected_id in base_ids:
            source = "h4_base"

        elif selected_id in extra_ids:
            source = "strategic_extra"

        else:
            # This should normally never happen. Keeping it visible
            # makes diagnostics safer if the base class changes.
            source = "unknown"

        self._diag[
            "selected_source_counts"
        ][source] += 1

        self._diag[
            "strategy_source_counts"
        ][
            (
                strategy,
                source,
            )
        ] += 1

        # --------------------------------------------------------
        # Strategic-extra details
        # --------------------------------------------------------

        if source == "strategic_extra":

            self._diag[
                "selected_extra_strategy_counts"
            ][strategy] += 1

            action_type_name = (
                self._action_type_name(
                    selected_action
                )
            )

            self._diag[
                "selected_extra_action_type_counts"
            ][action_type_name] += 1

        # --------------------------------------------------------
        # Per-player index
        # --------------------------------------------------------

        player_index = getattr(
            state,
            "current_player",
            None,
        )

        player_stats = (
            self._diag[
                "by_player"
            ][player_index]
        )

        player_stats[
            "strategic_decisions"
        ] += 1

        player_stats[
            "strategy_counts"
        ][strategy] += 1

        player_stats[
            "selected_source_counts"
        ][source] += 1

        if source == "strategic_extra":
            player_stats[
                "selected_extra_strategy_counts"
            ][strategy] += 1

    # ============================================================
    # ACTION TYPE LABEL
    # ============================================================

    def _action_type_name(
        self,
        action,
    ):
        action_type = getattr(
            action,
            "action_type",
            None,
        )

        mapping = {
            ActionType.BUY_VISIBLE:
                "buy_visible",

            ActionType.BUY_RESERVED:
                "buy_reserved",

            ActionType.TAKE_GEMS:
                "take_gems",

            ActionType.RESERVE_VISIBLE:
                "reserve_visible",

            ActionType.RESERVE_TOP_DECK:
                "reserve_top_deck",
        }

        return mapping.get(
            action_type,
            str(
                action_type
            ),
        )

    # ============================================================
    # STRUCTURED STATS
    # ============================================================

    def get_diagnostic_stats(
        self,
    ):
        """
        Return a plain-Python snapshot suitable for printing,
        logging, JSON conversion, or aggregation.
        """

        decisions = self._diag[
            "strategic_decisions"
        ]

        strategy_counts = dict(
            self._diag[
                "strategy_counts"
            ]
        )

        source_counts = dict(
            self._diag[
                "selected_source_counts"
            ]
        )

        extra_strategy_counts = dict(
            self._diag[
                "selected_extra_strategy_counts"
            ]
        )

        extra_action_counts = dict(
            self._diag[
                "selected_extra_action_type_counts"
            ]
        )

        strategy_source_counts = {
            f"{strategy}|{source}":
                count

            for (
                strategy,
                source,
            ), count

            in self._diag[
                "strategy_source_counts"
            ].items()
        }

        score_n = self._diag[
            "strategy_score_observations"
        ]

        rollout_n = self._diag[
            "selected_rollout_value_observations"
        ]

        by_player = {}

        for (
            player_index,
            player_stats,
        ) in self._diag[
            "by_player"
        ].items():

            by_player[
                player_index
            ] = {
                "strategic_decisions":
                    player_stats[
                        "strategic_decisions"
                    ],

                "strategy_counts":
                    dict(
                        player_stats[
                            "strategy_counts"
                        ]
                    ),

                "selected_source_counts":
                    dict(
                        player_stats[
                            "selected_source_counts"
                        ]
                    ),

                "selected_extra_strategy_counts":
                    dict(
                        player_stats[
                            "selected_extra_strategy_counts"
                        ]
                    ),
            }

        return {
            "total_select_action_calls":
                self._diag[
                    "total_select_action_calls"
                ],

            "forced_node_calls":
                self._diag[
                    "forced_node_calls"
                ],

            "forced_node_type_counts":
                dict(
                    self._diag[
                        "forced_node_type_counts"
                    ]
                ),

            "overflow_events":
                self._diag[
                    "overflow_events"
                ],

            "overflow_preceding_action_counts":
                dict(
                    self._diag[
                        "overflow_preceding_action_counts"
                    ]
                ),

            "overflow_start_token_counts":
                dict(
                    self._diag[
                        "overflow_start_token_counts"
                    ]
                ),

            "average_overflow_start_token_count":
                (
                    self._diag[
                        "overflow_start_token_sum"
                    ]
                    / self._diag[
                        "overflow_start_token_observations"
                    ]
                    if self._diag[
                        "overflow_start_token_observations"
                    ]
                    else 0.0
                ),

            "overflow_size_counts":
                dict(
                    self._diag[
                        "overflow_size_counts"
                    ]
                ),

            "average_overflow_size":
                (
                    self._diag[
                        "overflow_size_sum"
                    ]
                    / self._diag[
                        "overflow_events"
                    ]
                    if self._diag[
                        "overflow_events"
                    ]
                    else 0.0
                ),

            "overflow_gained_color_counts":
                dict(
                    self._diag[
                        "overflow_gained_color_counts"
                    ]
                ),

            "overflow_discarded_color_counts":
                dict(
                    self._diag[
                        "overflow_discarded_color_counts"
                    ]
                ),

            "overflow_events_rediscarding_gained_color":
                self._diag[
                    "overflow_events_rediscarding_gained_color"
                ],

            "overflow_rediscard_token_count":
                self._diag[
                    "overflow_rediscard_token_count"
                ],

            "overflow_rediscard_color_counts":
                dict(
                    self._diag[
                        "overflow_rediscard_color_counts"
                    ]
                ),

            "strategic_decisions":
                decisions,

            "strategy_counts":
                strategy_counts,

            "strategy_percentages":
                self._percent_dict(
                    strategy_counts,
                    decisions,
                ),

            "selected_source_counts":
                source_counts,

            "selected_source_percentages":
                self._percent_dict(
                    source_counts,
                    decisions,
                ),

            "selected_extra_strategy_counts":
                extra_strategy_counts,

            "selected_extra_action_type_counts":
                extra_action_counts,

            "strategy_source_counts":
                strategy_source_counts,

            "extras_proposed_total":
                self._diag[
                    "extras_proposed_total"
                ],

            "average_extras_proposed_per_decision":
                (
                    self._diag[
                        "extras_proposed_total"
                    ]
                    / decisions
                    if decisions
                    else 0.0
                ),

            "decisions_with_any_extra":
                self._diag[
                    "decisions_with_any_extra"
                ],

            "decisions_with_any_extra_pct":
                self._pct(
                    self._diag[
                        "decisions_with_any_extra"
                    ],
                    decisions,
                ),

            "average_noble_score":
                (
                    self._diag[
                        "noble_score_sum"
                    ]
                    / score_n
                    if score_n
                    else 0.0
                ),

            "average_reserve_score":
                (
                    self._diag[
                        "reserve_score_sum"
                    ]
                    / score_n
                    if score_n
                    else 0.0
                ),

            "average_selected_rollout_value":
                (
                    self._diag[
                        "selected_rollout_value_sum"
                    ]
                    / rollout_n
                    if rollout_n
                    else 0.0
                ),

            "by_player":
                by_player,
        }

    # ============================================================
    # HUMAN-READABLE SUMMARY
    # ============================================================

    def format_diagnostic_summary(
        self,
    ):
        stats = (
            self.get_diagnostic_stats()
        )

        decisions = stats[
            "strategic_decisions"
        ]

        strategy_counts = stats[
            "strategy_counts"
        ]

        strategy_pct = stats[
            "strategy_percentages"
        ]

        source_counts = stats[
            "selected_source_counts"
        ]

        source_pct = stats[
            "selected_source_percentages"
        ]

        lines = []

        lines.append(
            "=== HeuristicAgent12 Diagnostics ==="
        )

        lines.append(
            f"Strategic decisions: {decisions}"
        )

        lines.append(
            f"Forced-node calls: {stats['forced_node_calls']}"
        )

        lines.append("Forced-node split:")

        forced_counts = stats.get(
            "forced_node_type_counts",
            {},
        )

        for node_type_name in (
            "OVERFLOW_DISCARD",
            "NOBLE_CLAIM",
        ):
            count = forced_counts.get(
                node_type_name,
                0,
            )

            pct = (
                100.0
                * count
                / stats["forced_node_calls"]
                if stats["forced_node_calls"]
                else 0.0
            )

            lines.append(
                f"  {node_type_name}: "
                f"{count} "
                f"({pct:.2f}%)"
            )

        other_forced = sum(
            count
            for name, count
            in forced_counts.items()
            if name not in (
                "OVERFLOW_DISCARD",
                "NOBLE_CLAIM",
            )
        )

        if other_forced:
            lines.append(
                f"  OTHER: {other_forced}"
            )

        overflow_events = stats.get(
            "overflow_events",
            0,
        )

        lines.append("")
        lines.append("Overflow episode details:")
        lines.append(
            f"  Episodes: {overflow_events}"
        )
        lines.append(
            "  Previous action: "
            f"{stats.get('overflow_preceding_action_counts', {})}"
        )
        lines.append(
            "  Starting token count: "
            f"{stats.get('overflow_start_token_counts', {})}"
        )
        lines.append(
            "  Average starting tokens: "
            f"{stats.get('average_overflow_start_token_count', 0.0):.3f}"
        )
        lines.append(
            "  Overflow size: "
            f"{stats.get('overflow_size_counts', {})}"
        )
        lines.append(
            "  Average overflow size: "
            f"{stats.get('average_overflow_size', 0.0):.3f}"
        )
        lines.append(
            "  Gained colors before overflow: "
            f"{stats.get('overflow_gained_color_counts', {})}"
        )
        lines.append(
            "  Discarded colors: "
            f"{stats.get('overflow_discarded_color_counts', {})}"
        )

        rediscard_events = stats.get(
            "overflow_events_rediscarding_gained_color",
            0,
        )

        rediscard_pct = (
            100.0
            * rediscard_events
            / overflow_events
            if overflow_events
            else 0.0
        )

        lines.append(
            "  Immediately re-discarded a gained color: "
            f"{rediscard_events} "
            f"({rediscard_pct:.2f}%)"
        )
        lines.append(
            "  Re-discarded gained tokens: "
            f"{stats.get('overflow_rediscard_token_count', 0)}"
        )
        lines.append(
            "  Re-discarded colors: "
            f"{stats.get('overflow_rediscard_color_counts', {})}"
        )

        lines.append("")
        lines.append("Strategy selected:")

        for strategy in (
            self.NOBLE,
            self.RESERVE,
            self.NEUTRAL,
        ):
            lines.append(
                "  "
                f"{strategy}: "
                f"{strategy_counts.get(strategy, 0)} "
                f"({strategy_pct.get(strategy, 0.0):.2f}%)"
            )

        lines.append("")
        lines.append("Final selected action source:")

        for source in (
            "h4_base",
            "strategic_extra",
            "unknown",
        ):
            count = source_counts.get(
                source,
                0,
            )

            if (
                count == 0
                and source == "unknown"
            ):
                continue

            lines.append(
                "  "
                f"{source}: "
                f"{count} "
                f"({source_pct.get(source, 0.0):.2f}%)"
            )

        lines.append("")
        lines.append(
            "Selected strategic extras by strategy:"
        )

        for strategy, count in sorted(
            stats[
                "selected_extra_strategy_counts"
            ].items()
        ):
            lines.append(
                f"  {strategy}: {count}"
            )

        lines.append("")
        lines.append(
            "Selected strategic extras by action type:"
        )

        extra_action_counts = stats[
            "selected_extra_action_type_counts"
        ]

        if extra_action_counts:
            for action_type, count in sorted(
                extra_action_counts.items()
            ):
                lines.append(
                    f"  {action_type}: {count}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.append("")
        lines.append(
            "Strategic-extra availability:"
        )

        lines.append(
            "  "
            f"Decisions with >=1 extra: "
            f"{stats['decisions_with_any_extra']} "
            f"({stats['decisions_with_any_extra_pct']:.2f}%)"
        )

        lines.append(
            "  "
            f"Average extras proposed / decision: "
            f"{stats['average_extras_proposed_per_decision']:.3f}"
        )

        lines.append("")
        lines.append(
            "Average classifier scores:"
        )

        lines.append(
            "  "
            f"Noble: "
            f"{stats['average_noble_score']:.3f}"
        )

        lines.append(
            "  "
            f"Reserve: "
            f"{stats['average_reserve_score']:.3f}"
        )

        lines.append("")
        lines.append(
            "Strategy x selected source:"
        )

        for strategy in (
            self.NOBLE,
            self.RESERVE,
            self.NEUTRAL,
        ):
            base_count = stats[
                "strategy_source_counts"
            ].get(
                f"{strategy}|h4_base",
                0,
            )

            extra_count = stats[
                "strategy_source_counts"
            ].get(
                f"{strategy}|strategic_extra",
                0,
            )

            lines.append(
                "  "
                f"{strategy}: "
                f"H4={base_count}, "
                f"extra={extra_count}"
            )

        return "\n".join(
            lines
        )

    # ============================================================
    # SMALL HELPERS
    # ============================================================

    def _pct(
        self,
        numerator,
        denominator,
    ):
        if not denominator:
            return 0.0

        return (
            100.0
            * numerator
            / denominator
        )

    def _percent_dict(
        self,
        counts,
        denominator,
    ):
        return {
            key:
                self._pct(
                    value,
                    denominator,
                )

            for key, value
            in counts.items()
        }
