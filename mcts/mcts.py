import random 
from splendor_v1.mcts.node import Node

import math
import time 
class MCTS:

    def __init__(self, simulations=1000):

        self.simulations = simulations

    def search(self, env, state, return_root=False):

        # Player whose decision we are trying to improve
        root_player = state.current_player

        # Root node starts from the current game state
        root = Node(
            state=state.clone(),
            untried_actions=env._legal_actions(state)
        )

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


        print(
            f"select={selection_time:.4f}s "
            f"expand={expansion_time:.4f}s "
            f"rollout={rollout_time:.4f}s "
            f"backup={backup_time:.4f}s"
        )

        print("Legal root actions:", len(root.untried_actions) + len(root.children))
        print("Expanded children:", len(root.children))
        print(
            [(child.visits, child.value) for child in root.children]
        )

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


        action = node.untried_actions.pop()

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