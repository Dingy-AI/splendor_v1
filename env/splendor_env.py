import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SplendorEnv(gym.Env):
    def __init__ (self):
        #TODO
        
        #I need to define the action space to allow for more expansions in the future
        #Pick Gems (6 gem types) -> Red, Blue, Green, Black, White
        #Pick 3 -
        #Pick 2 -

        #Buy a card
        # Reserve a card
        self.action_space = spaces.Discrete(512)
        #

        return None 
    
    def reset(self, seed=420):
        #TODO
        obs = None
        info = None
        return obs, info
    
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
    
