import random 
from splendor_v1.mcts.node import Node

import math

class MCTS:

    def __init__(self, env, simulations=1000):

        self.env = env

        self.simulations = simulations

    def search(self, state):

        root = Node(
            state=state.clone()
        )

        for _ in range(self.simulations):


            try:
                leaf = self.select(root)

                child = self.expand(leaf)

                value = self.random_rollout(
                    self.env,
                    child.state
                )

                self.backup(child, value)


            except Exception as e:
                print(f"MCTS simulation failed: {e}")
                continue

        return self.best_action(root)

    def ucb_score(
        self,
        parent,
        child,
        exploration_constant=math.sqrt(2)
    ) -> float:

        # Every child should be explored at least once
        if child.visits == 0:
            return float("inf")

        exploitation = child.value_sum / child.visits

        exploration = exploration_constant * math.sqrt(
            math.log(parent.visits) / child.visits
        )

        return exploitation + exploration



    def select(self, node):

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
                    child
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

    def random_rollout(self, env, child, return_state=False):

        rollout_env = env.clone()
        rollout_child = child.state.clone()
        rollout_env.state = rollout_child
        root_player = rollout_env.state.current_player

        terminated = False
        steps = 0

        while not terminated:

            actions = rollout_env._legal_actions(rollout_env.state)

            if not actions:

                return -1
                # raise RuntimeError(
                #     "Rollout reached state with no legal actions"
                # )

                 

            action = random.choice(actions)

            obs, reward, terminated, truncated, info = rollout_env.step(action)
            steps += 1

            if steps > 500:
                terminated = True
                return 0.0

        winners = info['winners']

        if return_state:
            return rollout_env.state

        return 1.0 if root_player in winners else 0.0


    def backup(self):

        pass