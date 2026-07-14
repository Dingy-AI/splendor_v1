import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, CardColor

@pytest.fixture
def env():
    return SplendorEnv()

def test_legal_Actions(env):
    state = env.reset()




    actions = env._legal_actions(env.state)

    # print(env.state.visible_cards[0][2])
    # print(env.state.visible_cards[0][2].cost[CardColor.WHITE])
    # print(env.state.visible_cards[0][2].cost[CardColor.RED])
    # print(env.state.visible_cards[0][2].cost[CardColor.BLACK])
    # print(env.state.visible_cards[0][2].cost[CardColor.BLUE])
    # print(env.state.visible_cards[0][2].cost[CardColor.GREEN])

    # print(actions[0].action_type)
