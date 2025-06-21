from isaaclab.app import AppLauncher

# Launch Isaac Sim in GUI (non-headless) mode
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Import your environment and configuration
from source.isaaclab_tasks.isaaclab_tasks.direct.humanoid.humanoid_env import HumanoidEnv
from source.isaaclab_tasks.isaaclab_tasks.direct.humanoid.humanoid_cfg import HumanoidEnvCfg

import torch

# Create the configuration instance
cfg = HumanoidEnvCfg()

# Create the environment
env = HumanoidEnv(cfg)

# Set the joint positions to the default joint pose (static, no control applied)
env.robot.write_joint_position_to_sim(env.robot.data.default_joint_pos.clone())

# Start the rendering loop (no physics simulation step)
while simulation_app.is_running():
    env.sim.render()
