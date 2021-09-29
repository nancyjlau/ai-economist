import numpy as np

ENV_CONF_DEFAULT = {
            # ===== STANDARD ARGUMENTS ======
            "n_agents": 4,  # Number of non-planner agents
            "world_size": [15, 15],  # [Height, Width] of the env world
            "episode_length": 1000,  # Number of time-steps per episode
            # In multi-action-mode, the policy selects an action for each action
            # subspace (defined in component code)
            # Otherwise, the policy selects only 1 action
            "multi_action_mode_agents": False,
            "multi_action_mode_planner": False,
            # When flattening observations, concatenate scalar & vector observations
            # before output
            # Otherwise, return observations with minimal processing
            "flatten_observations": False,
            # When Flattening masks, concatenate each action subspace mask
            # into a single array
            # Note: flatten_masks = True is recommended for masking action logits
            "flatten_masks": False,
            # ===== COMPONENTS =====
            # Which components to use
            "components": [
                # (1) Building houses
                {"SingleAgentJob": {"time":40, "week_hrs":2}},
                # (2) Trading collectible resources
                #{"ContinuousDoubleAuction": {"max_num_orders": 5}},
                # (3) Movement and resource collection
                # {"Gather": {}},
            ],
            # ===== SCENARIO =====
            # Which scenario class to use
            # (optional) kwargs of the chosen scenario class
            "scenario_name": "uniform/simple_wood_and_stone",
            # (optional) kwargs of the chosen scenario class
            "starting_agent_coin": 10,
            "starting_stone_coverage": 0.10,
            "starting_wood_coverage": 0.10,
}

ENV_CONF_COMMUNISM = {
    'components': [
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        ('Gather', {}),
        ('PeriodicBracketTax', {
            'bracket_spacing': "us-federal",
            'period': 100,
            'tax_model': 'fixed-bracket-rates',
            # Communism
            'fixed_bracket_rates': np.ones(7).tolist(),
        }),
    ],
}
ENV_CONF_MACHINE = {
    'scenario_name': 'PlannerLikeAgentsEnv',
    # only productivity is rewarded
    'mixing_weight_gini_vs_coin': 1,
}

ENV_CONF_DYSTOPIA = {
    'scenario_name': 'PlannerLikeAgentsEnv',
    # weights equality and productivity equally
    'mixing_weight_gini_vs_coin': 0,
}
