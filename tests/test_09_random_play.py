import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START, DISCARD_COLORS
from copy import deepcopy


import random
@pytest.fixture
def env():
    return SplendorEnv()

#default is 1000
@pytest.mark.parametrize("seed", range(10))
def test_random_rollout(seed, env):

    random.seed(seed)

    env.reset(seed=seed)

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):

        terminated = env._check_terminated(env.state)

        if terminated:
            break

        legal_actions = env._legal_actions(env.state)

        assert len(legal_actions) > 0, (
            f"No legal actions in nonterminal state.\n"
            f"Seed: {seed}\n"
            f"Step: {steps}\n"
            f"Node type: {env.state.node_type}\n"
            f"Current player: {env.state.current_player}"
        )

        action = random.choice(legal_actions)

        try:
            _, _, terminated, truncated, _ = env.step(action)
        except Exception as e:
            pytest.fail(
                f"Exception during random rollout.\n"
                f"Seed: {seed}\n"
                f"Step: {steps}\n"
                f"Action: {action}\n"
                f"Exception: {e}"
            )

        steps += 1

        assert steps < 500, (
            f"Game exceeded maximum step count.\n"
            f"Seed: {seed}"
        )
# def test_random_game_complete(env):
#     random.seed(99)

#     env.reset()

#     done = False

#     steps = 0 

#     while not done:


#         actions = env._legal_actions(env.state)

#         count_buy_visible = 0
#         count_buy_reserves = 0
#         for action in actions:
#             if action.action_type == ActionType.BUY_RESERVED:
#                 count_buy_reserves += 1
#             if action.action_type == ActionType.BUY_VISIBLE:
#                 count_buy_visible += 1
#         print("START HERE:")
#         print("count buy visible:", count_buy_visible, "count buy reserves", count_buy_reserves)

#         print('current player', env.state.current_player)

#         print('bank', env.state.bank)

#         print("steps count:", steps)
#         print('player reserve:')
#         print(env.state.players[0].reserved_cards)
#         print(env.state.players[1].reserved_cards)

#         print("player gems")
#         print(env.state.players[0].gems)
#         print(env.state.players[1].gems)

#         print("player bonuses")
#         print(env.state.players[0].bonuses)
#         print(env.state.players[1].bonuses)

#         print('visible cards 1')
#         print(env.state.visible_cards[1])
#         print('visible cards 2')

#         print(env.state.visible_cards[2])
#         print('visible cards 3')

#         print(env.state.visible_cards[3])
#         print('action list', actions)
#         print(env.state.players[0].points)

#         assert len(actions) > 0
#         action = random.choice(actions)

#         print("current action")
#         print(action)


#         _, _, terminated, truncated, _ = env.step(action)        

#         # print("RIGHT AFTER ACTION CHECK")
#         for color in DISCARD_COLORS:
#             if color != GemColor.GOLD:
#                 if env.state.players[0].gems[color] + env.state.players[1].gems[color] + env.state.bank[color] != 4:
#                     raise ValueError(
#                     f"Invalid gem sum {color} and {env.state.players[0].gems[color] + env.state.players[1].gems[color] + env.state.bank[color]} "
#                 )

#             else:
#                 if env.state.players[0].gems[color] + env.state.players[1].gems[color] + env.state.bank[color] != 5:
#                     f"Invalid gem sum {color} and {env.state.players[0].gems[color] + env.state.players[1].gems[color] + env.state.bank[color]} "


#         done = terminated or truncated

#         steps += 1

#         assert steps < 500  # safety check        


# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[11] - Failed: Exception during random rollout. 
# REASON/FIXED -> Tier 1 deck runs out and legal buy visible tries to for loop through it

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[33] - AssertionError: No legal actions.
# REASON/SADNESS -> actually ran out of moves from reserving tier 2 cards and then having a 'bad' combination of gems

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[42] - Failed: Exception during random rollout.
# FIXED

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[48] - Failed: Exception during random rollout.
# FIXED

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[49] - Failed: Exception during random rollout.
# FIXED

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[51] - AssertionError: No legal actions.
# REASON -> Ran out of moves

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[59] - Failed: Exception during random rollout.
# FIXED

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[66] - AssertionError: No legal actions.
# REASON -> Ran out of moves

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[75] - Failed: Exception during random rollout.
# FIXED

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[82] - Failed: Exception during random rollout.
# FIXED

# FAILED splendor_v1/tests/test_09_random_play.py::test_random_rollout[99] - AssertionError: No legal actions.
# REASON -> Ran out of moves
