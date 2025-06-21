# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Direct-v0",
    entry_point="isaaclab_tasks.direct.humanoid.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.humanoid.humanoid_cfg:HumanoidEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.humanoid.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.humanoid.agents.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": "isaaclab_tasks.direct.humanoid.agents:skrl_ppo_cfg.yaml",
    },
)
