import random 
from splendor_v1.mcts.node import Node
from collections import Counter, defaultdict
import math
import time
import numpy as np
from splendor_v1.mcts.rollout import random_rollout, heuristic_rollout, heuristic_rollout_v2
from splendor_v1.mcts.neural_evaluator import neural_evaluate, slow_neural_evaluate
import torch
class MCTS:

    def __init__(self, 
                 simulations=25, 
                 rollout_type="random",
                 selection_type='ucb',
                 model=None):
        self.rollout_type = rollout_type
        self.simulations = simulations
        self.model = model
        self.selection_type = selection_type


        if (rollout_type == "neural" or selection_type == 'puct') and model is None:
            raise ValueError(
                "A model is required for neural rollout."
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

        for _ in range(self.simulations):

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

                    # Terminal / dead-end node
                    if value is None:
                        value = 0.0

                    else:
                        # Neural value is from the
                        # perspective of the player
                        # to move at node.state.
                        if (
                            node.state.current_player
                            != root_player
                        ):
                            value = -value

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

        # -------------------------
        # 5. Choose final action
        # -------------------------
        if not root.children:
            return None

        best_child = max(
            root.children,
            key=lambda child: child.visits
        )

        if return_root:
            return best_child.action, root
        return best_child.action


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

                child = max(
                    parent.children,
                    key=lambda child: self.puct_score(
                        parent,
                        child,
                        root_player,
                    ),
                )

                # Lazily create the child state
                # only when PUCT actually selects it.
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
        prior = 0.0

        if self.selection_type == "puct":

            policy_probs, _ = neural_evaluate(
                env,
                self.model,
                node.state,
            )

            action_id = env.action_to_id(
                action
            )

            prior = policy_probs[
                action_id
            ].item()

        child = Node(
            state=child_state,
            parent=node,
            action=action,
            untried_actions=child_env._legal_actions(
                child_state
            ),
            prior=prior,
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

                if root_player in child.state.winners:
                    return 1.0

                if len(child.state.winners) == 0:
                    return 0.0

                return -1.0

            legal_actions = self.get_legal_actions(
                env,
                child
            )

            if not legal_actions:
                # Deadlock
                return 0.0                


            policy_probs, value = neural_evaluate(
                env,
                self.model,
                child.state,
                legal_actions=legal_actions
            )

            # Neural value is from the perspective
            # of the player to move at child.state.
            if child.state.current_player != root_player:
                value = -value

            return value

        raise ValueError(
            f"Unknown rollout type: {self.rollout_type}"
        )

    def puct_score(
        self,
        parent,
        child,
        root_player,
        c_puct=3,
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
            c_puct
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
        teacher_mode=False
    ):

        if env._check_terminated(node.state):
            node.expanded = True
            return

        legal_actions = self.get_legal_actions(
            env,
            node
        )

        if not legal_actions:
            node.state.game_over = True
            node.state.winners = []
            node.expanded = True
            return

        if teacher_mode:

            # Strong teacher:
            # uniform policy + no neural value
            legal_probs = torch.full(
                (len(legal_actions),),
                1.0 / len(legal_actions),
                dtype=torch.float32,
            )

            value = 0.0

        else:

            # Normal neural MCTS
            legal_probs, value = neural_evaluate(
                env,
                self.model,
                node.state,
                legal_actions=legal_actions,
            )
        for action, prior in zip(
            legal_actions,
            legal_probs,
        ):

            child = Node(
                state=None,
                parent=node,
                action=action,
                prior=prior.item(),
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
        alpha=0.3,
        epsilon=0.25,
    ):
        if not root.children:
            return

        noise = np.random.dirichlet(
            [alpha] * len(root.children)
        )

        for child, n in zip(root.children, noise):
            child.prior = (
                (1.0 - epsilon) * child.prior
                + epsilon * n
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