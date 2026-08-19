import random 
from splendor_v1.mcts.node import Node
from collections import Counter, defaultdict
import math
import time 
class MCTS:

    def __init__(self, simulations=1000):

        self.simulations = simulations

    def search(self, env, state, return_root=False, debug=False):

        # Player whose decision we are trying to improve
        root_player = state.current_player

        # Root node starts from the current game state
        root = Node(
            state=state.clone(),
            untried_actions=env._legal_actions(state)
        )

        initial_legal_count = len(root.untried_actions)


        selection_time = 0
        expansion_time = 0
        rollout_time = 0
        backup_time = 0


        for _ in range(self.simulations):

            # -------------------------
            # 1. Selection
            # -------------------------
            start = time.perf_counter()
            leaf = self.select(root, root_player)
            selection_time += time.perf_counter() - start

            # -------------------------
            # 2. Expansion
            # -------------------------
            start = time.perf_counter()
            if leaf.untried_actions:
                child = self.expand(env, leaf)
            else:
                child = leaf
            expansion_time += time.perf_counter() - start

            # -------------------------
            # 3. Rollout
            # -------------------------
            start = time.perf_counter()
            value = self.random_rollout(
                env,
                child,
                root_player=root_player
            )
            rollout_time += time.perf_counter() - start

            # -------------------------
            # 4. Backup
            # -------------------------
            start = time.perf_counter()
            self.backup(
                child,
                value
            )
            backup_time += time.perf_counter() - start

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

        # -------------------------
        # 7. Diagnostics
        # -------------------------
        if debug:

            expanded_count = len(root.children)

            revisited_count = sum(
                1
                for child in root.children
                if child.visits > 1
            )

            max_visits = max(
                child.visits
                for child in root.children
            )

            min_visits = min(
                child.visits
                for child in root.children
            )

            total_child_visits = sum(
                child.visits
                for child in root.children
            )

            print("\nMCTS Search Diagnostics")
            print("-----------------------")

            print(f"Turn: {state.turn_number}")
            print(f"Root player: {root_player}")
            print(f"Simulations: {self.simulations}")

            print()
            print(f"Initial legal actions: {initial_legal_count}")
            print(f"Expanded children: {expanded_count}")
            print(f"Unexpanded actions: {len(root.untried_actions)}")
            print(f"Revisited children: {revisited_count}")

            print()
            print(f"Root visits: {root.visits}")
            print(f"Total child visits: {total_child_visits}")
            print(f"Max child visits: {max_visits}")
            print(f"Min child visits: {min_visits}")

            print()
            print("Timing:")
            print(f"  Selection: {selection_time:.4f}s")
            print(f"  Expansion: {expansion_time:.4f}s")
            print(f"  Rollout:   {rollout_time:.4f}s")
            print(f"  Backup:    {backup_time:.4f}s")

            print()
            print("Children:")

            children_sorted = sorted(
                root.children,
                key=lambda child: child.visits,
                reverse=True
            )

            for i, child in enumerate(children_sorted):

                average_value = (
                    child.value / child.visits
                    if child.visits > 0
                    else 0.0
                )

                print(
                    f"{i:>2}. "
                    f"visits={child.visits:<4} "
                    f"value_sum={child.value:<7.2f} "
                    f"avg={average_value:<7.3f} "
                    f"action={child.action}"
                )


            type_counts = Counter(
                child.action.action_type
                for child in root.children
            )

            print()
            print("Expanded Children by Action Type")
            print("--------------------------------")

            for action_type, count in type_counts.items():
                print(
                    f"{action_type.name:<20} "
                    f"{count}"
                )



            print()
            print("Visits by Action Type")
            print("---------------------")

            groups = defaultdict(list)

            for child in root.children:
                groups[child.action.action_type].append(child)

            for action_type, children in groups.items():

                total_visits = sum(
                    child.visits
                    for child in children
                )

                average_visits = (
                    total_visits / len(children)
                )

                average_value = sum(
                    child.value / child.visits
                    for child in children
                    if child.visits > 0
                ) / len(children)

                print(
                    f"{action_type.name:<20} "
                    f"children={len(children):<4} "
                    f"visits={total_visits:<4} "
                    f"avg_visits={average_visits:.2f} "
                    f"avg_value={average_value:.3f}"
                )


        if return_root:
            return best_child.action, root
        return best_child.action


    # def ucb_score(
    #     self,
    #     parent,
    #     child,
    #     exploration_constant=math.sqrt(2)
    # ) -> float:

    #     # Every child should be explored at least once
    #     if child.visits == 0:
    #         return float("inf")

    #     exploitation = child.value / child.visits

    #     exploration = exploration_constant * math.sqrt(
    #         math.log(parent.visits) / child.visits
    #     )

    #     return exploitation + exploration

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

    def select(self, node, root_player):

        current = node

        while True:

            # If this node still has actions we haven't expanded,
            # stop here. Expansion should happen next.
            if current.untried_actions:
                return current

            # Terminal / dead-end node
            if not current.children:
                return current

            # Otherwise move down the most promising branch
            current = max(
                current.children,
                key=lambda child: self.ucb_score(
                    current,
                    child,
                    root_player
                )
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

        child = Node(
            state=child_state,
            parent=node,
            action=action,
            untried_actions=child_env._legal_actions(child_state),

        )

        node.children.append(child)

        return child
    

    def random_rollout(
        self,
        env,
        child,
        root_player,
        return_state=False
    ):

        rollout_state = child.state.clone()

        steps = 0

        while True:

            actions = env._legal_actions(rollout_state)

            if not actions:
                return -1.0

            action = random.choice(actions)

            obs, reward, terminated, truncated, info = env.step(
                action,
                state=rollout_state
            )

            steps += 1

            if terminated or truncated:
                break

            if steps >= 500:
                return 0.0

        if return_state:
            return rollout_state

        winners = info["winners"]

        return 1.0 if root_player in winners else 0.0


    def backup(self, node, value):

        current = node

        while current is not None:

            current.visits += 1
            current.value += value

            current = current.parent