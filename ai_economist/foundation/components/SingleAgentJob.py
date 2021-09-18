# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import math
from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)
import ai_economist.foundation.base.base_env as glob_env

@component_registry.add
class SingleAgentJob (BaseComponent):
    """
    Allows mobile agnets to decide to whether to work on a project based on the
    time and the deadline of the project.

    """
    name= "SingleAgentJob"
    component_type= "SingleAgentJob"
    required_entities = ["Time", "Project","Skill"]
    agent_subclasses = ["BasicMobileAgent"]
    def __init__(
        self,
        *base_component_args,
         time,
         week_hrs,
        skill_dist="none",
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.time = int(time)
        assert self.time >= 0

        # self.project = int(project)
        # assert self.project >= 1

        # self.skill_dist = skill_dist.lower()
        # assert self.skill_dist in ["none"]

        # self.payment = int(payment)
        # assert self.payment >= 0.0
        self.total_project = []
        self.skill_dist = skill_dist
        self.week_hrs = week_hrs

    def can_agent_claim_project(self,agent,action):
        
        # action_mask = [0]*((40//self.week_hrs)*4)
        # if agent.state["endogenous"]["Timecommitment"][1] != 0:
        #     temp = 40 - agent.state["endogenous"]["Timecommitment"][1]
        # else:
        #     temp=0
        
        # for i in range(0,len(action_mask)):
        #     if (i%20)*self.week_hrs <= temp :
        #         action_mask[i] = 0

        if self.action_mask [action] == 0:
            return True
        else:
            return False
 
    
    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        # This component adds 1 action that mobile agents can take: build a house
        if agent_cls_name == "BasicMobileAgent":
            return ((40//self.week_hrs)*4)

        return None
    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"project_payment": float(10), "project_skill": 1}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood to house+coin for agents that choose to build and can.
        """
        world = self.world
        
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Build! (If you can.)
            elif action <= 80: #
                    max_time = 40/self.week_hrs
                    action_project = math.ceil(action/20)
                    for i in world.project_board.projectList.keys():
                        if world.project_board.projectList[i]["claimed"] == -1 and action_project == world.project_board.projectList[i]["hardness"]:
                            project_time = world.project_board.projectList[i]["project_time"]/agent.state["endogenous"]["Skill"]
                            if self.can_agent_claim_project(agent,action):
                                action = max_time if action%(max_time) == 0 else action%max_time
                                timecommit_len = len(agent.state["endogenous"]["Timecommitment"])-1
                                if (project_time//action > timecommit_len):
                                    agent.state["endogenous"]["Timecommitment"].extend([0]*(int(project_time//action)-timecommit_len))
                                j = 0
                                while(project_time >= 0):
                                    if action < project_time:
                                        agent.state["endogenous"]["Timecommitment"][j] += action*self.week_hrs
                                    else:
                                        agent.state["endogenous"]["Timecommitment"][j] += project_time
                                    project_time -= action
                                    j += 1
                                world.project_board.projectList[i]["claimed"] = agent.idx
                                world.project_board.projectList[i]["agent_steps"] = j
                                world.project_board.project_count[action_project] -= 1
                                break
                                
                                
                # if self.project_done(agent):
                #     # Upskill the agent
                #     agent.state["endogenous"]["Skill"] *= agent.state["endogenous"]["Project_detail"].hardness
                #     # Receive payment for the project
                #     agent.state["inventory"]["Coin"] += agent.state["Payment"]
                #     project_complete.append(
                #         {
                #             "worker": agent.idx,
                #             "income": float(10),
                #             "skill_": float(agent.state["endogenous"]["Skill"])
                #         }
                #     )

            else:

                raise ValueError

       

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "coins": agent.state["endogenous"]["Coins"],
                "skill": agent.state["endogenous"]["Skill"],
                "masked_action": self.generate_masks(agent),
                "project_board": self.world.project_board.projectList
            }

        return obs_dict

    def generate_masks(self,agent):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """
        self.action_mask = [0]*((40//self.week_hrs)*4)
        if agent.state["endogenous"]["Timecommitment"][1] != 0:
            temp = 40 - agent.state["endogenous"]["Timecommitment"][1]
        else:
            temp= 40

        for i in range(0,len(self.action_mask)):
            if temp >=  (i%20)*self.week_hrs:
                self.action_mask[i] = 0
            else:
                self.action_mask[i] = 1

        
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        # for agent in self.world.agents:
        #     masks[agent.idx] = np.array([self.project_done(agent)])
        return self.action_mask

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
        for builds in self.builds:
            for build in builds:
                idx = build["builder"]
                build_stats[idx]["n_builds"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in build_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        num_houses = np.sum(world.maps.get("House") > 0)
        out_dict["total_builds"] = num_houses

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """
        world = self.world

        self.sampled_skills = {agent.idx: 1 for agent in world.agents}

        # PMSM = self.payment_max_skill_multiplier

        for agent in world.agents:
            if self.skill_dist == "none":
                sampled_skill = 1
                # pay_rate = 1
            # elif self.skill_dist == "pareto":
            #     sampled_skill = np.random.pareto(4)
            #     pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            # elif self.skill_dist == "lognormal":
            #     sampled_skill = np.random.lognormal(-1, 0.5)
            #     pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            else:
                raise NotImplementedError
            # TODO implent the payment calculation and upskill
            agent.state["build_payment"] = float(10)
            agent.state["build_skill"] = float(sampled_skill)

            self.sampled_skills[agent.idx] = sampled_skill

        self.builds = []

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.

        """
        return self.total_project

        


       

# @component_registry.add
# class Build(BaseComponent):
#     """
#     Allows mobile agents to build house landmarks in the world using stone and wood,
#     earning income.

#     Can be configured to include heterogeneous building skill where agents earn
#     different levels of income when building.

#     Args:
#         payment (int): Default amount of coin agents earn from building.
#             Must be >= 0. Default is 10.
#         payment_max_skill_multiplier (int): Maximum skill multiplier that an agent
#             can sample. Must be >= 1. Default is 1.
#         skill_dist (str): Distribution type for sampling skills. Default ("none")
#             gives all agents identical skill equal to a multiplier of 1. "pareto" and
#             "lognormal" sample skills from the associated distributions.
#         build_labor (float): Labor cost associated with building a house.
#             Must be >= 0. Default is 10.
#     """

#     name = "Build"
#     component_type = "Build"
#     # required_entities = ["Wood", "Stone", "Coin", "House", "Labor"]
#     required_entities = ["Time", "Project","Skill"]
#     agent_subclasses = ["BasicMobileAgent"]

#     def __init__(
#         self,
#         *base_component_args,   
#         resource_cost = {"Wood": 1, "Stone": 1},
#         skill_multiplier=1,
#         skill_dist="none",
#         build_labor=10.0,
#         **base_component_kwargs
#     ):
#         super().__init__(*base_component_args, **base_component_kwargs)

#         self.resource_cost = {"Wood": 1, "Stone": 1}

#         self.build_labor = float(build_labor)
#         assert self.build_labor >= 0

#         self.skill_dist = skill_dist.lower()
#         assert self.skill_dist in ["none", "pareto", "lognormal"]

#         self.sampled_skills = {}

#         self.builds = []

#     def agent_can_build(self, agent):
#         """Return True if agent can actually build in its current location."""
#         # See if the agent has the resources necessary to complete the action
#         for resource, cost in self.resource_cost.items():
#             if agent.state["inventory"][resource] < cost:
#                 return False

#         # Do nothing if this spot is already occupied by a landmark or resource
#         if self.world.location_resources(*agent.loc):
#             return False
#         # If we made it here, the agent can build.
#         return True

#     # Required methods for implementing components
#     # --------------------------------------------

#     def get_n_actions(self, agent_cls_name):
#         """
#         See base_component.py for detailed description.

#         Add a single action (build) for mobile agents.
#         """
#         # This component adds 1 action that mobile agents can take: build a house
#         if agent_cls_name == "BasicMobileAgent":
#             return 1

#         return None

#     def get_additional_state_fields(self, agent_cls_name):
#         """
#         See base_component.py for detailed description.

#         For mobile agents, add state fields for building skill.
#         """
#         if agent_cls_name not in self.agent_subclasses:
#             return {}
#         if agent_cls_name == "BasicMobileAgent":
#             return {"build_payment": float(self.payment), "build_skill": 1}
#         raise NotImplementedError

#     def component_step(self):
#         """
#         See base_component.py for detailed description.

#         Convert stone+wood to house+coin for agents that choose to build and can.
#         """
#         world = self.world
#         build = []
#         # Apply any building actions taken by the mobile agents
#         for agent in world.get_random_order_agents():

#             action = agent.get_component_action(self.name)

#             # This component doesn't apply to this agent!
#             if action is None:
#                 continue

#             # NO-OP!
#             if action == 0:
#                 pass

#             # Build! (If you can.)
#             elif action == 1:
#                 if self.agent_can_build(agent):
#                     # Remove the resources
#                     for resource, cost in self.resource_cost.items():
#                         agent.state["inventory"][resource] -= cost

#                     # Receive payment for the house
#                     agent.state["inventory"]["Coin"] += agent.state["build_payment"]

#                     # Incur the labor cost for building
#                     agent.state["endogenous"]["Labor"] += self.build_labor

#                     build.append(
#                         {
#                             "builder": agent.idx,
#                             "loc": np.array(agent.loc),
#                             "income": float(agent.state["build_payment"]),
#                         }
#                     )

#             else:
#                 raise ValueError

#         self.builds.append(build)

#     def generate_observations(self):
#         """
#         See base_component.py for detailed description.

#         Here, agents observe their build skill. The planner does not observe anything
#         from this component.
#         """

#         obs_dict = dict()
#         for agent in self.world.agents:
#             obs_dict[agent.idx] = {
#                 "build_payment": agent.state["build_payment"] / self.payment,
#                 "build_skill": self.sampled_skills[agent.idx],
#             }

#         return obs_dict

#     def generate_masks(self, completions=0):
#         """
#         See base_component.py for detailed description.

#         Prevent building only if a landmark already occupies the agent's location.
#         """

#         masks = {}
#         # Mobile agents' build action is masked if they cannot build with their
#         # current location and/or endowment
#         for agent in self.world.agents:
#             masks[agent.idx] = np.array([self.agent_can_build(agent)])

#         return masks

#     # For non-required customization
#     # ------------------------------

#     def get_metrics(self):
#         """
#         Metrics that capture what happened through this component.

#         Returns:
#             metrics (dict): A dictionary of {"metric_name": metric_value},
#                 where metric_value is a scalar.
#         """
#         world = self.world

#         build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
#         for builds in self.builds:
#             for build in builds:
#                 idx = build["builder"]
#                 build_stats[idx]["n_builds"] += 1

#         out_dict = {}
#         for a in world.agents:
#             for k, v in build_stats[a.idx].items():
#                 out_dict["{}/{}".format(a.idx, k)] = v

#         num_houses = np.sum(world.maps.get("House") > 0)
#         out_dict["total_builds"] = num_houses

#         return out_dict

#     def additional_reset_steps(self):
#         """
#         See base_component.py for detailed description.

#         Re-sample agents' building skills.
#         """
#         world = self.world

#         self.sampled_skills = {agent.idx: 1 for agent in world.agents}

#         PMSM = self.payment_max_skill_multiplier

#         for agent in world.agents:
#             if self.skill_dist == "none":
#                 sampled_skill = 1
#                 pay_rate = 1
#             elif self.skill_dist == "pareto":
#                 sampled_skill = np.random.pareto(4)
#                 pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
#             elif self.skill_dist == "lognormal":
#                 sampled_skill = np.random.lognormal(-1, 0.5)
#                 pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
#             else:
#                 raise NotImplementedError

#             agent.state["build_payment"] = float(pay_rate * self.payment)
#             agent.state["build_skill"] = float(sampled_skill)

#             self.sampled_skills[agent.idx] = sampled_skill

#         self.builds = []

#     def get_dense_log(self):
#         """
#         Log builds.

#         Returns:
#             builds (list): A list of build events. Each entry corresponds to a single
#                 timestep and contains a description of any builds that occurred on
#                 that timestep.

#         """
#         return self.builds
