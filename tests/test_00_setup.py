import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType

@pytest.fixture
def env():
    return SplendorEnv()


@pytest.fixture
def env_four():
    return SplendorEnv(4)

@pytest.fixture
def env_three():
    return SplendorEnv(3)

def test_env_exists(env):
    assert env is not None

def test_env_init_state(env):

    assert env.observation_encoder != None

    assert env.num_players == 2
    assert env.game_state == None
    assert env.seed == 420
    assert env.state == None


def test_env_three_reset(env_three):
    env_three.reset()
    assert len(env_three.state.players) == 3

def test_env_reset(env):
    env.reset()
    assert env.state != None
    assert env.state.node_type == NodeType.MAIN_DECISION
    assert len(env.state.players) == 2

    assert len(env.state.nobles) == 3
    assert env.state.bank[GemColor.WHITE] == 5
    assert env.state.bank[GemColor.GOLD] == 5

    # print(env.state.players)

    