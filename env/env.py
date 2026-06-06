import numpy as np
import gymnasium as gym
from gymnasium import spaces
from splendor_v1.env.core.actions import ACTION_SPACE_SIZE
from state import GameState

class SplendorEnv(gym.Env):
    def __init__ (self, num_players: int = 2, seed = 420):
        #TODO
        
        #I need to define the action space to allow for more expansions in the future
        #Pick Gems (6 gem types) -> Red, Blue, Green, Black, White
        #Pick 3 (9) - Red/Blue/Green, Red/Blue/Black, Red/Blue/White
        #             Red/Green/Black, Red/Green/White, Red/Black/White, Blue/Green/Black, Blue/Black/White, Green/Black/White
        #Pick 2 (5) - Red, Blue, Green, Black, White

        #Buy a card
        #Buy 1 of 12 (12) (15)

        # Reserve a card
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        # Reserve 1 of 15 (15)

        # 0-8 -> Pick 3
        # 9-13 -> Pick 2
        # 14-29 -> Buy a Card
        # 30-45 -> Reserve a Card
        # 46-48 -> Pick Noble
        # 49-61 -> Discard 13 #this will be a loop where we will discard 1 and recheck to discard another
        # 62+ -> Expansion
        #Total actions 9 + 5 + 15 + 15 = 61 Actions

        return None 
    
    
    def reset(self,num_players=2, seed=420,) -> GameState:
        #TODO
        obs = None
        info = None
        return obs, info
    
    def reset_deck(self):
        pass

    def create_visible_cards(self):
        pass

    def create_players(self, num_players):
        pass

    def create_nobles(self, num_players):
        pass 

    def step(self, action=None):
        #TODO 
        obs = None
        reward = None 
        terminated = None
        truncated = None
        info = None 
        return obs, reward, terminated, truncated, info

    def legal_actions(self):
        #TODO
        #write the mask that starts everything as false
        #and then sets the valid values to true

        return None
    def can_afford(self, card) -> bool:
        gold_needed = 0

        for color, cost in card.cost.items():
            effective_cost = max(
                0,
                cost - self.bonuses[color]
            )

            available = self.gems[color]

            if available < effective_cost:
                gold_needed += effective_cost - available

        return gold_needed <= self.gems[GemColor.GOLD]
    

    def get_reward(player_id = None):
        #TOOD
        return None 

    def clone(self):
        #TODO
        return None 

    def current_player(self):
        #TODO
        return None

    def is_terminal(self):
        #TODO
        return None
    
    def render(self):
        #TODO
        return None

    def get_winner(self):
        #TODO
        return None
    
