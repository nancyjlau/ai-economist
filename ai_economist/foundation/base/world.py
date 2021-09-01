# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import random 
from ai_economist.foundation.agents import agent_registry
from ai_economist.foundation.entities import resource_registry



class Projectboard:
    
    def __init__(self, size, n_agents, time_Cap, pay_Cap, hardness_Cap,step_Cap):
        self.size = size
        self.n_agents = n_agents
        self.timeCap = time_Cap
        self.payCap = pay_Cap
        self.hardnessCap = hardness_Cap
        self.stepCap = step_Cap
        self.projectList = {}
        i = 0
        while i < self.size:
            randTime = random.randint(1, time_Cap)
            randPay = random.randint(1, pay_Cap)
            randHard = random.randint(1, hardness_Cap)
            randstep = random.randint(1,step_Cap)            
            self.projectList.update({i:{'project_diff':randHard,'project_time':randTime,
                                        'steps':randstep,'agent_steps':0,
                                        'payment':randPay,'claimed':-1}})
            i += 1
            self.i = i
    
    def repopulate(self):
        while len(self.projectList) < self.size:
            randTime = random.randint(1, self.timeCap)
            randPay = random.randint(1, self.payCap)
            randHard = random.randint(1, self.hardnessCap)
            randstep = random.randint(1,self.stepCap)
            self.projectList.update({self.i:{'project_diff':randHard,'project_time':randTime,
                                        'steps':randstep,'agent_steps':0,
                                        'payment':randPay,'claimed':-1}})
            self.i += 1
            
            
    def generate_reward(self,rew):
        for i in self.projectList.keys():
            if self.projectList[i]["claimed"] != -1:
                if self.projectList[i]["steps"] !=0 and self.projectList[i]["agent_steps"] == 0:
                    rew[str(self.projectList[i]["claimed"])] = self.projectList[i]["payment"]
        return rew

    def step(self):
        for i in self.projectList.keys():
            if self.projectList[i]["claimed"] != -1:
                self.projectList[i]["agent_steps"] -= 1
                self.projectList[i]["steps"] -=1  
      
    def remove_projects(self):
        del_project_list = []
        for i in self.projectList.keys():
            if self.projectList[i]["claimed"] != -1:
                if self.projectList[i]["steps"] ==0:
                    del_project_list.append(i)

        for i in del_project_list:
            del self.projectList[i]
    
    def compound_step(self):
        self.step()
        self.remove_projects()
        self.repopulate()
    
class Maps:
    """Manages the spatial configuration of the world as a set of entity maps.

    A maps object is built during world construction, which is a part of environment
    construction. The maps object is accessible through the world object. The maps
    object maintains a map state for each of the spatial entities that are involved
    in the constructed environment (which are determined by the "required_entities"
    attributes of the Scenario and Component classes used to build the environment).

    The Maps class also implements some of the basic spatial logic of the game,
    such as which locations agents can occupy based on other agent locations and
    locations of various landmarks.

    Args:
        size (list): A length-2 list specifying the dimensions of the 2D world.
            Interpreted as [height, width].
        n_agents (int): The number of mobile agents (does not include planner).
        world_resources (list): The resources registered during environment
            construction.
        world_landmarks (list): The landmarks registered during environment
            construction.
    """

    def __init__(self, size, n_agents, world_resources):
        self.size = size
        self.sz_h, self.sz_w = size

        self.n_agents = n_agents

        self.resources = world_resources
        self.entities = world_resources

        self._maps = {}  # All maps
        self._blocked = []  # Solid objects that no agent can move through
        self._private = []  # Solid objects that only permit movement for parent agents
        self._public = []  # Non-solid objects that agents can move on top of
        self._resources = []  # Non-solid objects that can be collected

        self._private_landmark_types = []
        self._resource_source_blocks = []

        self._map_keys = []

        self._accessibility_lookup = {}

        for resource in self.resources:
            resource_cls = resource_registry.get(resource)
            if resource_cls.collectible:
                self._maps[resource] = np.zeros(shape=self.size)
                self._resources.append(resource)
                self._map_keys.append(resource)


        
        self._idx_map = np.stack(
            [i * np.ones(shape=self.size) for i in range(self.n_agents)]
        )
        self._idx_array = np.arange(self.n_agents)
        if self._accessibility_lookup:
            self._accessibility = np.ones(
                shape=[len(self._accessibility_lookup), self.n_agents] + self.size,
                dtype=np.bool,
            )
            self._net_accessibility = None
        else:
            self._accessibility = None
            self._net_accessibility = np.ones(
                shape=[self.n_agents] + self.size, dtype=np.bool
            )

        self._agent_locs = [None for _ in range(self.n_agents)]
        self._unoccupied = np.ones(self.size, dtype=np.bool)

    def clear(self, entity_name=None):
        """Clear resource and landmark maps."""
        if entity_name is not None:
            assert entity_name in self._maps
            if entity_name in self._private_landmark_types:
                self._maps[entity_name] = dict(
                    owner=-np.ones(shape=self.size, dtype=np.int16),
                    health=np.zeros(shape=self.size),
                )
            else:
                self._maps[entity_name] *= 0

        else:
            for name in self.keys():
                self.clear(entity_name=name)

        if self._accessibility is not None:
            self._accessibility = np.ones_like(self._accessibility)
            self._net_accessibility = None

    def clear_agent_loc(self, agent=None):
        """Remove agents or agent from the world map."""
        # Clear all agent locations
        if agent is None:
            self._agent_locs = [None for _ in range(self.n_agents)]
            self._unoccupied[:, :] = 1

        # Clear the location of the provided agent
        else:
            i = agent.idx
            if self._agent_locs[i] is None:
                return
            r, c = self._agent_locs[i]
            self._unoccupied[r, c] = 1
            self._agent_locs[i] = None

    def set_agent_loc(self, agent, r, c):
        """Set the location of agent to [r, c].

        Note:
            Things might break if you set the agent's location to somewhere it
            cannot access. Don't do that.
        """
        assert (0 <= r < self.size[0]) and (0 <= c < self.size[1])
        i = agent.idx
        # If the agent is currently on the board...
        if self._agent_locs[i] is not None:
            curr_r, curr_c = self._agent_locs[i]
            # If the agent isn't actually moving, just return
            if (curr_r, curr_c) == (r, c):
                return
            # Make the location the agent is currently at as unoccupied
            # (since the agent is going to move)
            self._unoccupied[curr_r, curr_c] = 1

        # Set the agent location to the specified coordinates
        # and update the occupation map
        agent.state["loc"] = [r, c]
        self._agent_locs[i] = [r, c]
        self._unoccupied[r, c] = 0

    def keys(self):
        """Return an iterable over map keys."""
        return self._maps.keys()

    def values(self):
        """Return an iterable over map values."""
        return self._maps.values()

    def items(self):
        """Return an iterable over map (key, value) pairs."""
        return self._maps.items()

    def get(self, entity_name, owner=False):
        """Return the map or ownership for entity_name."""
        # assert entity_name in self._maps
        if entity_name in self._private_landmark_types:
            sub_key = "owner" if owner else "health"
            return self._maps[entity_name][sub_key]
        return self._maps[entity_name]

    def set(self, entity_name, map_state):
        """Set the map for entity_name."""
        if entity_name in self._private_landmark_types:
            assert "owner" in map_state
            assert self.get(entity_name, owner=True).shape == map_state["owner"].shape
            assert "health" in map_state
            assert self.get(entity_name, owner=False).shape == map_state["health"].shape

            h = np.maximum(0.0, map_state["health"])
            o = map_state["owner"].astype(np.int16)

            o[h <= 0] = -1
            tmp = o[h > 0]
            if len(tmp) > 0:
                assert np.min(tmp) >= 0

            self._maps[entity_name] = dict(owner=o, health=h)

            owned_by_agent = o[None] == self._idx_map
            owned_by_none = o[None] == -1
            self._accessibility[
                self._accessibility_lookup[entity_name]
            ] = np.logical_or(owned_by_agent, owned_by_none)
            self._net_accessibility = None

        else:
            assert self.get(entity_name).shape == map_state.shape
            self._maps[entity_name] = np.maximum(0, map_state)

            if entity_name in self._blocked:
                self._accessibility[
                    self._accessibility_lookup[entity_name]
                ] = np.repeat(map_state[None] == 0, self.n_agents, axis=0)
                self._net_accessibility = None

    def set_add(self, entity_name, map_state):
        """Add map_state to the existing map for entity_name."""
        assert entity_name not in self._private_landmark_types
        self.set(entity_name, self.get(entity_name) + map_state)

    def get_point(self, entity_name, r, c, **kwargs):
        """Return the entity state at the specified coordinates."""
        point_map = self.get(entity_name, **kwargs)
        return point_map[r, c]

    def set_point(self, entity_name, r, c, val, owner=None):
        """Set the entity state at the specified coordinates."""
        if entity_name in self._private_landmark_types:
            assert owner is not None
            h = self._maps[entity_name]["health"]
            o = self._maps[entity_name]["owner"]
            assert o[r, c] == -1 or o[r, c] == int(owner)
            h[r, c] = np.maximum(0, val)
            if h[r, c] == 0:
                o[r, c] = -1
            else:
                o[r, c] = int(owner)

            self._maps[entity_name]["owner"] = o
            self._maps[entity_name]["health"] = h

            self._accessibility[
                self._accessibility_lookup[entity_name], :, r, c
            ] = np.logical_or(o[r, c] == self._idx_array, o[r, c] == -1).astype(np.bool)
            self._net_accessibility = None

        else:
            self._maps[entity_name][r, c] = np.maximum(0, val)

            if entity_name in self._blocked:
                self._accessibility[
                    self._accessibility_lookup[entity_name]
                ] = np.repeat(np.array([val]) == 0, self.n_agents, axis=0)
                self._net_accessibility = None

    def set_point_add(self, entity_name, r, c, value, **kwargs):
        """Add value to the existing entity state at the specified coordinates."""
        self.set_point(
            entity_name,
            r,
            c,
            value + self.get_point(entity_name, r, c, **kwargs),
            **kwargs
        )

    def is_accessible(self, r, c, agent_id):
        """Return True if agent with id agent_id can occupy the location [r, c]."""
        return bool(self.accessibility[agent_id, r, c])

    def location_resources(self, r, c):
        """Return {resource: health} dictionary for any resources at location [r, c]."""
        return {
            k: self._maps[k][r, c] for k in self._resources if self._maps[k][r, c] > 0
        }


    @property
    def unoccupied(self):
        """Return a boolean map indicating which locations are unoccupied."""
        return self._unoccupied

    @property
    def accessibility(self):
        """Return a boolean map indicating which locations are accessible."""
        if self._net_accessibility is None:
            self._net_accessibility = self._accessibility.prod(axis=0).astype(np.bool)
        return self._net_accessibility

    @property
    def empty(self):
        """Return a boolean map indicating which locations are empty.

        Empty locations have no landmarks or resources."""
        return self.state.sum(axis=0) == 0

    @property
    def state(self):
        """Return the concatenated maps of landmark and resources."""
        return np.stack([self.get(k) for k in self.keys()]).astype(np.float32)

    @property
    def owner_state(self):
        """Return the concatenated ownership maps of private landmarks."""
        return np.stack(
            [self.get(k, owner=True) for k in self._private_landmark_types]
        ).astype(np.int16)

    @property
    def state_dict(self):
        """Return a dictionary of the map states."""
        return self._maps


class World:
    """Manages the environment's spatial- and agent-states.

    The world object represents the state of the environment, minus whatever state
    information is implicitly maintained by separate components. The world object
    maintains the spatial state through an instance of the Maps class. Agent states
    are maintained through instances of Agent classes (subclasses of BaseAgent),
    with one such instance for each of the agents in the environment.

    The world object is built during the environment construction, after the
    required entities have been registered. As part of the world object construction,
    it instantiates a map object and the agent objects.

    The World class adds some functionality for interfacing with the spatial state
    (the maps object) and setting/resetting agent locations. But its function is
    mostly to wrap the stateful, non-component environment objects.

    Args:
        world_size (list): A length-2 list specifying the dimensions of the 2D world.
            Interpreted as [height, width].
        n_agents (int): The number of mobile agents (does not include planner).
        world_resources (list): The resources registered during environment
            construction.
        world_landmarks (list): The landmarks registered during environment
            construction.
        multi_action_mode_agents (bool): Whether "mobile" agents use multi action mode
            (see BaseEnvironment in base_env.py).
        multi_action_mode_planner (bool): Whether the planner agent uses multi action
            mode (see BaseEnvironment in base_env.py).
    """

    def __init__(
        self,
        world_size,
        n_agents,
        world_resources,
        multi_action_mode_agents,
        multi_action_mode_planner,
    ):
        self.world_size = world_size
        self.n_agents = n_agents
        self.resources = world_resources
        self.multi_action_mode_agents = bool(multi_action_mode_agents)
        self.multi_action_mode_planner = bool(multi_action_mode_planner)
        self.maps = Maps(world_size, n_agents, world_resources)
        self.project_board = Projectboard(10,4,10,1000,4,100) 
        mobile_class = agent_registry.get("BasicMobileAgent")
        planner_class = agent_registry.get("BasicPlanner")
        self._agents = [
            mobile_class(i, multi_action_mode=self.multi_action_mode_agents)
            for i in range(self.n_agents)
        ]
        self._planner = planner_class(multi_action_mode=self.multi_action_mode_planner)

    @property
    def agents(self):
        """Return a list of the agent objects in the world (sorted by index)."""
        return self._agents

    @property
    def planner(self):
        """Return the planner agent object."""
        return self._planner

    @property
    def loc_map(self):
        """Return a map indicating the agent index occupying each location.

        Locations with a value of -1 are not occupied by an agent.
        """
        idx_map = -np.ones(shape=self.world_size, dtype=np.int16)
        for agent in self.agents:
            r, c = agent.loc
            idx_map[r, c] = int(agent.idx)
        return idx_map

    def get_random_order_agents(self):
        """The agent list in a randomized order."""
        agent_order = np.random.permutation(self.n_agents)
        agents = self.agents
        return [agents[i] for i in agent_order]

    def is_valid(self, r, c):
        """Return True if the coordinates [r, c] are within the game boundaries."""
        return (0 <= r < self.world_size[0]) and (0 <= c < self.world_size[1])

    def is_location_accessible(self, r, c, agent):
        """Return True if location [r, c] is accessible to agent."""
        if not self.is_valid(r, c):
            return False
        return self.maps.is_accessible(r, c, agent.idx)

    def can_agent_occupy(self, r, c, agent):
        """Return True if location [r, c] is accessible to agent and unoccupied."""
        if not self.is_location_accessible(r, c, agent):
            return False
        if self.maps.unoccupied[r, c]:
            return True
        return False

    def clear_agent_locs(self):
        """Take all agents off the board. Useful for resetting."""
        for agent in self.agents:
            agent.state["loc"] = [-1, -1]
        self.maps.clear_agent_loc()

    def agent_locs_are_valid(self):
        """Returns True if all agent locations comply with world semantics."""
        return all(
            [self.is_location_accessible(*agent.loc, agent) for agent in self.agents]
        )

    def set_agent_loc(self, agent, r, c):
        """Set the agent's location to coordinates [r, c] if possible.

        If agent cannot occupy [r, c], do nothing."""
        if self.can_agent_occupy(r, c, agent):
            self.maps.set_agent_loc(agent, r, c)
        return [int(coord) for coord in agent.loc]

    def location_resources(self, r, c):
        """Return {resource: health} dictionary for any resources at location [r, c]."""
        if not self.is_valid(r, c):
            return {}
        return self.maps.location_resources(r, c)

    def consume_resource(self, resource_name, r, c):
        """Consume a unit of resource_name from location [r, c]."""
        self.maps.set_point_add(resource_name, r, c, -1)
