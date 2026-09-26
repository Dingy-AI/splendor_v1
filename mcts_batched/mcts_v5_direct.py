"""MCTS V5: adaptive simulation pruning.

Forked from MCTS V4. V5.0 preserves the V4 tree policy and neural
evaluation semantics, but allocates search effort dynamically using
legal-action count, inherited root visits, and best-action stability.
No legal action is removed from the tree in this version.
"""

import random 

from splendor_v1.mcts.node import Node

from collections import Counter, defaultdict

import math

import time

import numpy as np

from splendor_v1.mcts.rollout import random_rollout, heuristic_rollout, heuristic_rollout_v2

from splendor_v1.mcts.neural_evaluator_v4 import slow_neural_evaluate

from splendor_v1.mcts_batched.direct_neural_evaluator import DirectNeuralEvaluator

import torch

class MCTS:



    def __init__(

        self,

        simulations=25,

        rollout_type="random",

        selection_type="ucb",

        model=None,

        evaluator=None,

        c_puct=3.0,

        dirichlet_alpha=0.3,

        dirichlet_epsilon=0.25,

        adaptive_simulations=True,

        min_simulations=80,

        check_interval=20,

        target_visits_per_action=20.0,

        single_action_simulations=4,

        stability_checks=3,

    ):



        self.rollout_type = rollout_type

        self.simulations = simulations

        self.model = model

        # --------------------------------------------------
        # Neural evaluator abstraction
        # --------------------------------------------------
        # Backward-compatible usage:
        #
        #     MCTS(model=model)
        #
        # automatically creates a DirectNeuralEvaluator.
        #
        # New injectable usage:
        #
        #     MCTS(evaluator=evaluator)
        #
        # lets future batched evaluators implement the same
        # evaluate(env, state, legal_actions) contract.
        if evaluator is None and model is not None:

            evaluator = DirectNeuralEvaluator(
                model=model,
            )

        self.evaluator = evaluator

        # Preserve direct model access when the evaluator exposes
        # it. This is only needed by the legacy slow diagnostic path.
        if self.model is None and evaluator is not None:

            self.model = getattr(
                evaluator,
                "model",
                None,
            )

        self.selection_type = selection_type



        self.c_puct = float(

            c_puct

        )



        self.dirichlet_alpha = float(

            dirichlet_alpha

        )



        self.dirichlet_epsilon = float(

            dirichlet_epsilon

        )



        # --------------------------------------------------
        # V5 adaptive-search / pruning configuration
        # --------------------------------------------------

        self.adaptive_simulations = bool(
            adaptive_simulations
        )

        self.min_simulations = max(
            1,
            int(min_simulations),
        )

        self.check_interval = max(
            1,
            int(check_interval),
        )

        self.target_visits_per_action = max(
            0.0,
            float(target_visits_per_action),
        )

        self.single_action_simulations = max(
            1,
            int(single_action_simulations),
        )

        self.stability_checks = max(
            1,
            int(stability_checks),
        )

        # Metadata from the most recent search. The replay
        # generator can copy this into rich replay samples.
        self.last_search_metadata = None



        self.rng = (

            np.random.default_rng()

        )



        if (
            rollout_type == "neural"
            or selection_type == "puct"
        ) and self.evaluator is None:

            raise ValueError(
                "A neural evaluator is required for neural "
                "rollout or PUCT selection. Pass either "
                "model=... or evaluator=... ."
            )



    def search(self, 

               env, 

               state, 

               root=None,

               return_root=False, 

               debug=False,

               add_root_noise=False,

               teacher_mode=False):



        # Player whose decision we are trying to improve

        root_player = state.current_player



        # --------------------------------------------------

        # 1. Create / restore root

        # --------------------------------------------------



        if root is None:

            root = Node(

                state=state.clone(),

            )



        else:

            root.parent = None



            if root.state is None:

                root.state = state.clone()



        # --------------------------------------------------

        # 2. Get legal actions from root

        # --------------------------------------------------



        legal_actions = self.get_legal_actions(

            env,

            root,

        )



        # --------------------------------------------------

        # 3. Handle terminal / dead-end

        # --------------------------------------------------



        if not legal_actions:



            print(

                "No legal actions:",

                "terminated=",

                env._check_terminated(state),

                "node_type=",

                state.node_type,

                "current_player=",

                state.current_player,

                "turn=",

                state.turn_number,

            )



            if return_root:

                return None, root



            state.game_over = True

            state.winners = []



            return None



        # selection_time = 0

        # expansion_time = 0

        # rollout_time = 0

        # backup_time = 0





        if (

            self.selection_type == "ucb"

            and root.untried_actions is None

        ):

            root.untried_actions = (

                legal_actions.copy()

            )





        root_noise_added = False



        # Existing/reused root

        if (

            add_root_noise

            and root.expanded

            and root.children

        ):

            self.add_dirichlet_noise(root)

            root_noise_added = True



        # --------------------------------------------------
        # V5 adaptive simulation budget
        # --------------------------------------------------

        initial_root_visits = int(
            root.visits
        )

        search_plan = self.build_search_plan(
            num_legal_actions=len(legal_actions),
            initial_root_visits=initial_root_visits,
        )

        simulations_run = 0
        best_action_history = []
        stop_reason = None
        latest_root_stats = None

        while simulations_run < search_plan["hard_limit"]:



            # -------------------------
            # 1. Selection
            # -------------------------
            # start = time.perf_counter()
            node = self.select(env, root, root_player)
            # selection_time += time.perf_counter() - start



            # -------------------------
            # 2. Expansion
            # -------------------------
            # start = time.perf_counter()
            if self.selection_type == "ucb":



                if node.untried_actions:

                    node = self.expand(

                        env,

                        node,

                    )



            elif self.selection_type == "puct":



                if not node.expanded:



                    value = self.expand_all_with_priors(

                        env,

                        node,

                        root_player=root_player,

                        teacher_mode=teacher_mode

                    )



                    # Fresh root was just expanded
                    if (

                        add_root_noise

                        and node is root

                        and not root_noise_added

                        and root.children

                    ):

                        self.add_dirichlet_noise(root)

                        root_noise_added = True



                else:

                    # This should normally only happen
                    # for terminal/dead-end expanded nodes
                    # returned by selection.
                    value = self.rollout(

                        env,

                        node,

                        root_player=root_player,

                    )

            # expansion_time += time.perf_counter() - start



            # -------------------------
            # 3. Rollout
            # -------------------------
            # start = time.perf_counter()
            if self.selection_type != "puct":



                value = self.rollout(

                        env,

                        node,

                        root_player=root_player

                )



            # rollout_time += time.perf_counter() - start



            # -------------------------
            # 4. Backup
            # -------------------------



            # start = time.perf_counter()
            self.backup(

                node,

                value

            )
            # backup_time += time.perf_counter() - start

            simulations_run += 1

            # Single-action roots have no policy choice to
            # resolve. We still search a few simulations so
            # the retained subtree receives some value/search
            # information for tree reuse.
            if (
                self.adaptive_simulations
                and len(legal_actions) == 1
                and simulations_run >= search_plan["soft_budget"]
            ):
                stop_reason = "single_legal_action"
                break

            should_check = (
                simulations_run % self.check_interval == 0
                or simulations_run == search_plan["soft_budget"]
                or simulations_run == search_plan["hard_limit"]
            )

            if not should_check:
                continue

            latest_root_stats = self.calculate_root_search_stats(
                env=env,
                root=root,
                remaining_simulations=(
                    search_plan["hard_limit"]
                    - simulations_run
                ),
            )

            best_action_key = latest_root_stats[
                "best_action_key"
            ]

            if best_action_key is not None:
                best_action_history.append(
                    best_action_key
                )

            stable_checks = self.trailing_stability_count(
                best_action_history
            )

            # Fixed mode is deliberately identical to V4:
            # always consume the full requested simulation
            # budget.
            if not self.adaptive_simulations:
                continue

            # The soft budget is the earliest point at which
            # ordinary multi-action searches may terminate.
            if simulations_run < search_plan["soft_budget"]:
                continue

            if stable_checks >= self.stability_checks:
                stop_reason = "stable_after_soft_budget"
                break

        if stop_reason is None:
            stop_reason = (
                "fixed_budget"
                if not self.adaptive_simulations
                else "max_simulations"
            )

        # Capture final root statistics even if the last search
        # ended between normal check intervals (for example a
        # single-action root).
        latest_root_stats = self.calculate_root_search_stats(
            env=env,
            root=root,
            remaining_simulations=max(
                0,
                search_plan["hard_limit"]
                - simulations_run,
            ),
        )

        stable_checks = self.trailing_stability_count(
            best_action_history
        )

        self.last_search_metadata = {
            "adaptive_search": self.adaptive_simulations,
            "max_simulations": int(self.simulations),
            "min_simulations": int(
                min(self.min_simulations, self.simulations)
            ),
            "check_interval": int(self.check_interval),
            "target_visits_per_action": float(
                self.target_visits_per_action
            ),
            "single_action_simulations": int(
                self.single_action_simulations
            ),
            "stability_checks_required": int(
                self.stability_checks
            ),
            "num_legal_actions": int(len(legal_actions)),
            "initial_root_visits": int(initial_root_visits),
            "target_total_root_visits": int(
                search_plan["target_total_root_visits"]
            ),
            "soft_budget_simulations": int(
                search_plan["soft_budget"]
            ),
            "actual_simulations": int(simulations_run),
            "final_root_visits": int(root.visits),
            "stop_reason": stop_reason,
            "best_action_stability_checks": int(
                stable_checks
            ),
            "root_noise_enabled": bool(add_root_noise),
            **latest_root_stats,
        }

        # Attach metadata to the root when possible. This is
        # convenient for replay generators that already retain
        # the returned root. last_search_metadata remains the
        # authoritative fallback.
        try:
            root.search_metadata = dict(
                self.last_search_metadata
            )
        except (AttributeError, TypeError):
            pass

        # -------------------------

        # 5. Choose final action

        # -------------------------

        if not root.children:

            if return_root:

                return None, root



            return None



        best_child = max(

            root.children,

            key=lambda child: child.visits

        )



        if return_root:

            return best_child.action, root

        return best_child.action





    def build_search_plan(
        self,
        num_legal_actions,
        initial_root_visits,
    ):
        """Build the V5 search budget for one root.

        The requested ``simulations`` value remains the hard
        maximum. Adaptive search only reduces work below that
        ceiling; it never exceeds the V4 budget.
        """

        hard_limit = max(
            1,
            int(self.simulations),
        )

        if not self.adaptive_simulations:
            return {
                "hard_limit": hard_limit,
                "soft_budget": hard_limit,
                "target_total_root_visits": (
                    initial_root_visits
                    + hard_limit
                ),
            }

        if num_legal_actions <= 1:
            soft_budget = min(
                hard_limit,
                self.single_action_simulations,
            )

            return {
                "hard_limit": soft_budget,
                "soft_budget": soft_budget,
                "target_total_root_visits": (
                    initial_root_visits
                    + soft_budget
                ),
            }

        target_total_root_visits = int(
            math.ceil(
                self.target_visits_per_action
                * num_legal_actions
            )
        )

        required_new_simulations = max(
            0,
            target_total_root_visits
            - int(initial_root_visits),
        )

        effective_minimum = min(
            self.min_simulations,
            hard_limit,
        )

        soft_budget = max(
            effective_minimum,
            required_new_simulations,
        )

        soft_budget = min(
            soft_budget,
            hard_limit,
        )

        # Ordinary searches are inspected in fixed-size chunks.
        # Rounding up means we never stop *before* the target
        # search effort due only to chunk boundaries.
        soft_budget = self.round_up_to_interval(
            soft_budget,
            self.check_interval,
            hard_limit,
        )

        return {
            "hard_limit": hard_limit,
            "soft_budget": soft_budget,
            "target_total_root_visits": (
                target_total_root_visits
            ),
        }



    @staticmethod
    def round_up_to_interval(
        value,
        interval,
        maximum,
    ):
        if value <= 0:
            return min(interval, maximum)

        rounded = (
            (value + interval - 1)
            // interval
            * interval
        )

        return min(
            int(rounded),
            int(maximum),
        )



    @staticmethod
    def trailing_stability_count(
        action_history,
    ):
        if not action_history:
            return 0

        latest = action_history[-1]
        count = 0

        for action_key in reversed(
            action_history
        ):
            if action_key != latest:
                break

            count += 1

        return count



    def action_key(
        self,
        env,
        action,
    ):
        """Return a stable identifier for root-action tracking."""

        try:
            return int(
                env.action_to_id(action)
            )
        except Exception:
            # This is diagnostic/stability metadata only. The
            # canonical action id is preferred, but repr keeps
            # the search usable for non-Splendor test envs.
            return repr(action)



    def calculate_root_search_stats(
        self,
        env,
        root,
        remaining_simulations,
    ):
        """Measure root confidence without changing the tree.

        V5.0 uses only best-action stability for early stopping.
        The other statistics are intentionally recorded now so
        later pruning revisions can be driven by replay evidence.
        """

        if not root.children:
            return {
                "best_action_key": None,
                "best_action_visits": 0,
                "second_action_visits": 0,
                "top1_visit_share": 0.0,
                "top2_visit_share": 0.0,
                "top3_visit_share": 0.0,
                "visited_action_fraction": 0.0,
                "normalized_visit_entropy": 0.0,
                "visit_margin": 0,
                "visit_margin_ratio": 0.0,
                "winner_locked": False,
                "best_action_q": 0.0,
                "second_action_q": 0.0,
                "q_gap": 0.0,
                "network_prior_entropy": 0.0,
            }

        by_visits = sorted(
            root.children,
            key=lambda child: child.visits,
            reverse=True,
        )

        visits = [
            max(0, int(child.visits))
            for child in by_visits
        ]

        total_visits = sum(visits)
        num_actions = len(by_visits)

        best_child = by_visits[0]
        second_child = (
            by_visits[1]
            if num_actions >= 2
            else None
        )

        best_visits = visits[0]
        second_visits = (
            visits[1]
            if num_actions >= 2
            else 0
        )

        if total_visits > 0:
            visit_probs = [
                visit / total_visits
                for visit in visits
                if visit > 0
            ]

            entropy = -sum(
                p * math.log(p)
                for p in visit_probs
            )

            if num_actions > 1:
                normalized_entropy = (
                    entropy
                    / math.log(num_actions)
                )
            else:
                normalized_entropy = 0.0
        else:
            normalized_entropy = 0.0

        visited_action_fraction = (
            sum(visit > 0 for visit in visits)
            / num_actions
        )

        top1_share = (
            best_visits / total_visits
            if total_visits > 0
            else 0.0
        )

        top2_share = (
            sum(visits[:2]) / total_visits
            if total_visits > 0
            else 0.0
        )

        top3_share = (
            sum(visits[:3]) / total_visits
            if total_visits > 0
            else 0.0
        )

        visit_margin = (
            best_visits
            - second_visits
        )

        visit_margin_ratio = (
            visit_margin / total_visits
            if total_visits > 0
            else 0.0
        )

        winner_locked = (
            num_actions == 1
            or best_visits
            > second_visits
            + max(0, int(remaining_simulations))
        )

        best_q = (
            best_child.value / best_child.visits
            if best_child.visits > 0
            else 0.0
        )

        second_q = (
            second_child.value / second_child.visits
            if (
                second_child is not None
                and second_child.visits > 0
            )
            else 0.0
        )

        # Prefer raw network priors. Fall back to current priors
        # for older/non-neural nodes that do not carry them.
        raw_priors = []

        for child in root.children:
            prior = getattr(
                child,
                "network_prior",
                None,
            )

            if prior is None:
                prior = getattr(
                    child,
                    "prior",
                    0.0,
                )

            raw_priors.append(
                max(0.0, float(prior))
            )

        prior_sum = sum(raw_priors)

        if prior_sum > 0.0:
            prior_probs = [
                prior / prior_sum
                for prior in raw_priors
                if prior > 0.0
            ]

            prior_entropy = -sum(
                p * math.log(p)
                for p in prior_probs
            )

            if num_actions > 1:
                prior_entropy /= math.log(
                    num_actions
                )
            else:
                prior_entropy = 0.0
        else:
            prior_entropy = 0.0

        return {
            "best_action_key": self.action_key(
                env,
                best_child.action,
            ),
            "best_action_visits": int(
                best_visits
            ),
            "second_action_visits": int(
                second_visits
            ),
            "top1_visit_share": float(
                top1_share
            ),
            "top2_visit_share": float(
                top2_share
            ),
            "top3_visit_share": float(
                top3_share
            ),
            "visited_action_fraction": float(
                visited_action_fraction
            ),
            "normalized_visit_entropy": float(
                normalized_entropy
            ),
            "visit_margin": int(
                visit_margin
            ),
            "visit_margin_ratio": float(
                visit_margin_ratio
            ),
            "winner_locked": bool(
                winner_locked
            ),
            "best_action_q": float(
                best_q
            ),
            "second_action_q": float(
                second_q
            ),
            "q_gap": float(
                best_q - second_q
            ),
            "network_prior_entropy": float(
                prior_entropy
            ),
        }



    def ucb_score(

        self,

        parent,

        child,

        root_player,

        exploration_constant=1.414

    ):



        if child.visits == 0:

            return float("inf")



        average_value = child.value/ child.visits



        exploration = exploration_constant * math.sqrt(

            math.log(parent.visits) / child.visits

        )



        # Root player's turn:

        # choose outcomes that are good for root

        if parent.state.current_player == root_player:

            exploitation = average_value



        # Opponent's turn:

        # choose outcomes that are bad for root

        else:

            exploitation = -average_value



        return exploitation + exploration



    def select(

        self,

        env,

        node,

        root_player,

    ):



        current = node



        while True:



            if self.selection_type == "ucb":



                # UCB stops when there are still

                # unexpanded actions at this node.

                if current.untried_actions:

                    return current



                # Terminal / dead-end node

                if not current.children:

                    return current



                current = max(

                    current.children,

                    key=lambda child: self.ucb_score(

                        current,

                        child,

                        root_player,

                    ),

                )



            elif self.selection_type == "puct":



                # PUCT stops when this node

                # has not been expanded yet.

                if not current.expanded:

                    return current



                # Terminal / dead-end node

                if not current.children:

                    return current



                parent = current



                child = self.select_puct_child(

                    parent,

                    root_player,

                )



                self.materialize_state(

                    env,

                    child,

                )



                current = child



            else:

                raise ValueError(

                    f"Unknown selection type: "

                    f"{self.selection_type}"

                )



    def expand(self, env, node):



        if len(node.untried_actions) == 0:

            return None





        action = random.choice(node.untried_actions)

        node.untried_actions.remove(action)



        # Clone environment so parent/game isn't modified

        child_env = env.clone()



        # Start the cloned environment from this node's state

        child_env.state = node.state.clone()



        # Apply action

        child_env.step(action)



        # Fetch resulting state

        child_state = child_env.state



        # child = Node(

        #     state=child_state,

        #     parent=node,

        #     action=action,

        #     untried_actions=child_env._legal_actions(child_state),



        # )



        # node.children.append(child)



        # return child



        child = Node(

            state=child_state,

            parent=node,

            action=action,

            untried_actions=child_env._legal_actions(

                child_state

            ),

        )



        node.children.append(

            child

        )



        return child







    def backup(self, node, value):



        current = node



        while current is not None:



            current.visits += 1

            current.value += value



            current = current.parent



    def rollout(

        self,

        env,

        child,

        root_player,

    ):



        self.materialize_state(

            env,

            child,

        )



        if self.rollout_type == "random":

            return random_rollout(

                env,

                child,

                root_player,

            )



        if self.rollout_type == "heuristic":

            return heuristic_rollout(

                env,

                child,

                root_player,

            )



        if self.rollout_type == "heuristic_v2":

            return heuristic_rollout_v2(

                env,

                child,

                root_player,

            )



        if self.rollout_type == "neural":





            if env._check_terminated(child.state):



                # WDL semantics:

                #   sole winner -> +1

                #   shared result / no winner -> 0

                #   sole loser -> -1

                if len(child.state.winners) != 1:

                    return 0.0



                if root_player == child.state.winners[0]:

                    return 1.0



                return -1.0



            legal_actions = self.get_legal_actions(

                env,

                child

            )



            if not legal_actions:

                # Deadlock

                return 0.0                





            policy_probs, value = self.evaluator.evaluate(

                env=env,

                state=child.state,

                legal_actions=legal_actions,

            )



            # Neural value is from the perspective

            # of the player to move at child.state.

            if child.state.current_player != root_player:

                value = -value



            return value



        raise ValueError(

            f"Unknown rollout type: {self.rollout_type}"

        )



    def select_puct_child(

        self,

        parent,

        root_player,

    ):

        if not parent.children:

            raise ValueError(

                "Cannot select from a node without children."

            )



        # Calculate once for this parent comparison.

        sqrt_parent_visits = math.sqrt(

            max(parent.visits, 1)

        )

        same_player = (

            parent.state.current_player == root_player

        )



        best_child = None

        best_score = 0.0



        for child in parent.children:

            visits = child.visits



            average_value = (

                child.value / visits

                if visits != 0

                else 0.0

            )



            exploitation = (

                average_value

                if same_player

                else -average_value

            )



            exploration = (

                self.c_puct

                * child.prior

                * sqrt_parent_visits

                / (1 + visits)

            )

            score = exploitation + exploration



            # Like max(), keep the first child when scores tie.

            if best_child is None or score > best_score:

                best_child = child

                best_score = score



        return best_child









    def puct_score(

        self,

        parent,

        child,

        root_player,

    ):



        if child.visits == 0:

            average_value = 0.0

        else:

            average_value = (

                child.value / child.visits

            )



        if parent.state.current_player == root_player:

            exploitation = average_value

        else:

            exploitation = -average_value



        exploration = (

            self.c_puct

            * child.prior

            * math.sqrt(

                max(parent.visits, 1)

            )

            / (1 + child.visits)

        )



        return exploitation + exploration



    def slow_expand_all_with_priors(

        self,

        env,

        node,

    ):



        if env._check_terminated(node.state):

            node.expanded = True

            return





        legal_actions = env._legal_actions(

            node.state

        )



        if not legal_actions:

            node.state.game_over = True

            node.state.winners = []

            node.expanded = True            

            return





        if self.model is None:

            raise RuntimeError(
                "slow_expand_all_with_priors requires direct "
                "access to the underlying model."
            )

        policy_probs, _ = slow_neural_evaluate(

            env,

            self.model,

            node.state,

            legal_actions=legal_actions

        )



        for action in legal_actions:



            # child_state = node.state.clone()



            # env.step(

            #     action,

            #     state=child_state,

            # )



            action_id = env.action_to_id(

                action

            )



            child = Node(

                # state=child

                state=None,

                parent=node,

                action=action,

                prior=policy_probs[

                    action_id

                ].item(),

            )



            node.children.append(

                child

            )



        node.expanded = True





    def expand_all_with_priors(

        self,

        env,

        node,

        root_player=None,

        teacher_mode=False

    ):



        if env._check_terminated(node.state):



            node.expanded = True



            return self.terminal_value(

                node.state,

                root_player,

            )



        legal_actions = self.get_legal_actions(

            env,

            node

        )



        if not legal_actions:

            node.state.game_over = True

            node.state.winners = []

            node.expanded = True

            return 0.0



        if teacher_mode:



            # Strong teacher:

            # uniform policy + no neural value

            legal_probs = torch.full(

                (len(legal_actions),),

                1.0 / len(legal_actions),

                dtype=torch.float32,

            )



            # No neural value.

            # Instead estimate the leaf by playing out the game.

            value = self.rollout(

                env,

                node,

                root_player=root_player,

            )



        else:



            # Normal neural MCTS

            legal_probs, value = self.evaluator.evaluate(

                env=env,

                state=node.state,

                legal_actions=legal_actions,

            )



            # neural value is relative to

            # player-to-move

            if (

                node.state.current_player

                != root_player

            ):

                value = -value





        prior_values = legal_probs.detach().cpu().tolist()





        for action, prior in zip(

            legal_actions,

            prior_values,

        ):



            child = Node(

                state=None,

                parent=node,

                action=action,

                prior=prior,

            )



            # Preserve the original policy prior before

            # Dirichlet noise can modify child.prior.

            child.network_prior = float(

                prior

            )



            node.children.append(

                child

            )



        node.expanded = True



        return value





    def materialize_state(

        self,

        env,

        node,

    ):

        if node.state is not None:

            return



        if node.parent is None:

            raise ValueError(

                "Cannot materialize root node "

                "without a state."

            )



        if node.parent.state is None:

            raise ValueError(

                "Cannot materialize child because "

                "parent state is missing."

            )



        child_state = node.parent.state.clone()



        env.step(

            node.action,

            state=child_state,

        )



        node.state = child_state



    def flip_tree_values(

        self,

        node,

    ):



        node.value = -node.value



        for child in node.children:

            self.flip_tree_values(

                child

            )





    def get_legal_actions(

        self,

        env,

        node,

    ):



        if node.legal_actions is not None:

            return node.legal_actions



        if node.state is None:

            raise ValueError(

                "Cannot generate legal actions "

                "for a node without a state."

            )



        node.legal_actions = env._legal_actions(

            node.state

        )



        return node.legal_actions



    def add_dirichlet_noise(

        self,

        root,

    ):



        if not root.children:

            return



        noise = self.rng.dirichlet(

            [

                self.dirichlet_alpha

            ]

            * len(root.children)

        )



        for child, n in zip(

            root.children,

            noise,

        ):



            child.prior = (

                (

                    1.0

                    - self.dirichlet_epsilon

                )

                * child.prior

                +

                self.dirichlet_epsilon

                * n

            )



    def print_root_debug(

        self,

        env,

        root,

        chosen_action,

        top_k=5,

    ):

        if not root.children:

            print("Root has no children.")

            return



        total_visits = sum(

            child.visits

            for child in root.children

        )



        print("\n" + "=" * 80)



        print(

            f"Turn: {root.state.turn_number} | "

            f"Player: {root.state.current_player} | "

            f"Root visits: {root.visits}"

        )



        print(f"Chosen action: {chosen_action}")



        # --------------------------------------------------

        # Chosen action

        # --------------------------------------------------



        chosen_child = next(

            (

                child

                for child in root.children

                if child.action == chosen_action

            ),

            None,

        )



        if chosen_child is not None:



            visit_fraction = (

                chosen_child.visits / total_visits

                if total_visits > 0

                else 0.0

            )



            average_value = (

                chosen_child.value / chosen_child.visits

                if chosen_child.visits > 0

                else 0.0

            )



            print(

                "\nChosen action statistics:"

            )



            print(

                f"  Prior: {chosen_child.prior:.4f}"

            )



            print(

                f"  Visits: {chosen_child.visits}"

            )



            print(

                f"  Visit %: {visit_fraction:.4f}"

            )



            print(

                f"  Q value: {average_value:.4f}"

            )



        # --------------------------------------------------

        # Highest neural priors

        # --------------------------------------------------



        print(

            f"\nTop {top_k} by NETWORK PRIOR:"

        )



        by_prior = sorted(

            root.children,

            key=lambda child: child.prior,

            reverse=True,

        )



        for rank, child in enumerate(

            by_prior[:top_k],

            start=1,

        ):



            visit_fraction = (

                child.visits / total_visits

                if total_visits > 0

                else 0.0

            )



            print(

                f"{rank}. "

                f"{child.action} | "

                f"P={child.prior:.4f} | "

                f"N={child.visits} | "

                f"Visit%={visit_fraction:.4f}"

            )



        # --------------------------------------------------

        # Highest MCTS visits

        # --------------------------------------------------



        print(

            f"\nTop {top_k} by MCTS VISITS:"

        )



        by_visits = sorted(

            root.children,

            key=lambda child: child.visits,

            reverse=True,

        )



        for rank, child in enumerate(

            by_visits[:top_k],

            start=1,

        ):



            visit_fraction = (

                child.visits / total_visits

                if total_visits > 0

                else 0.0

            )



            average_value = (

                child.value / child.visits

                if child.visits > 0

                else 0.0

            )



            print(

                f"{rank}. "

                f"{child.action} | "

                f"N={child.visits} | "

                f"Visit%={visit_fraction:.4f} | "

                f"P={child.prior:.4f} | "

                f"Q={average_value:.4f}"

            )



        print("=" * 80)





    def terminal_value(

        self,

        state,

        root_player,

    ):



        # WDL semantics:

        #   sole winner -> +1

        #   shared result / no winner -> 0

        #   sole loser -> -1

        if len(state.winners) != 1:

            return 0.0



        if root_player == state.winners[0]:

            return 1.0



        return -1.0
