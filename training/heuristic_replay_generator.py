import pickle
import numpy as np
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

class HeuristicReplayGenerator:

    def __init__(
        self,
        env,
        agent,
        replay_buffer,
    ):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer


    def generate_game(self, seed=None):

        # -------------------------
        # RESET
        # -------------------------

        if seed is None:
            result = self.env.reset()
        else:
            result = self.env.reset(seed=seed)

        # Adjust this if your reset API differs.
        if hasattr(result, "current_player"):
            state = result
        else:
            state = self.env.state

        trajectory = []

        terminated = False

        # -------------------------
        # PLAY GAME
        # -------------------------

        while not terminated:

            player = state.current_player

            # IMPORTANT:
            # use the exact same encoder that your
            # neural-network self-play currently uses.
            observation = self.env.observation_encoder.encoder(
                state
            )

            policy = self.agent.get_policy(
                self.env,
                state,
                action_size=ACTION_SPACE_SIZE,
            )

            trajectory.append(
                (
                    observation.copy(),
                    policy.copy(),
                    player,
                )
            )

            action = self.agent.select_action(
                self.env,
                state,
            )

            if action is None:
                raise RuntimeError(
                    "Heuristic returned None "
                    "before termination."
                )

            result = self.env.step(action)

            # Gymnasium-style
            if isinstance(result, tuple):
                if len(result) == 5:
                    _, _, terminated, truncated, _ = result

                    terminated = (
                        terminated
                        or truncated
                    )

                elif len(result) == 4:
                    _, _, terminated, _ = result

            state = self.env.state

        # -------------------------
        # ASSIGN FINAL VALUES
        # -------------------------

        winners = state.winners

        for observation, policy, player in trajectory:

            if player in winners:
                value = 1.0
            else:
                value = -1.0

            self.replay_buffer.add(
                (
                    observation,
                    policy,
                    value,
                )
            )

        return len(trajectory)