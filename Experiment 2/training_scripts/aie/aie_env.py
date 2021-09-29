import numpy as np
from ai_economist import foundation
from ai_economist.foundation.base.base_env import BaseEnvironment
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
import logging
from aie.env_conf import ENV_CONF_DEFAULT

'''
world-map
0: stone
1: wood
2: house
3: water
4: stone (available ??)
5: wood (available ??)
6: wall

world-idx_map
0: house (mine: 1, others: player_idx + 2)
1: player's location (mine: 1, others: player_idx + 2)
'''
OBS_SPACE_AGENT = spaces.Dict({
    'skill': spaces.Discrete(5),
    'project_count': spaces.Box(0, 2, shape=(4,),dtype=np.int64),
    'masked_action': spaces.Box(0, 1, shape=(80,),dtype=np.int64),
    
})
ACT_SPACE_AGENT = spaces.Discrete(80)


class AIEEnv(MultiAgentEnv):
    def __init__(self, env_config, force_dense_logging: bool = False):
        self.env: BaseEnvironment = foundation.make_env_instance(**{
            **ENV_CONF_DEFAULT,
            **env_config,
        })
        
        self.observation_space = OBS_SPACE_AGENT
        self.action_space = ACT_SPACE_AGENT
        self.force_dense_logging = force_dense_logging
        

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset(force_dense_logging=self.force_dense_logging)
        obs = {
            k: {
                k1: v1 if type(v1) is np.ndarray else np.array([v1])
                for k1, v1 in v.items()
                if k1 in OBS_SPACE_AGENT.spaces.keys()
            }
            for k, v in obs.items()
        }
        logging.debug("Reached Rest")
        return obs

    def step(self, actions: MultiAgentDict) -> (MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict):
        
        obs, r, done, info = self.env.step(actions)
        obs = {
            k: {
                k1: v1 if type(v1) is np.ndarray else np.array([v1])
                for k1, v1 in v.items()
                if k1 in OBS_SPACE_AGENT.spaces.keys()
            }
            for k, v in obs.items()
        }
        
        return obs, r, done, info
