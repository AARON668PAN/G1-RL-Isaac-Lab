from __future__ import annotations

from isaaclab_assets import HUMANOID_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass




@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 23
    observation_space = 46
    state_space = 0
    clip_actions = 100.

    root_body = "pelvis"

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    joint_gears: list = [
    67.5,  # left_hip_pitch_joint
    67.5,  # right_hip_pitch_joint
    67.5,  # torso_joint
    67.5,  # left_hip_roll_joint
    67.5,  # right_hip_roll_joint
    67.5,  # left_shoulder_pitch_joint
    67.5,  # right_shoulder_pitch_joint
    67.5,  # left_hip_yaw_joint
    67.5,  # right_hip_yaw_joint
    67.5,  # left_shoulder_roll_joint
    67.5,  # right_shoulder_roll_joint
    67.5,  # left_shoulder_yaw_joint
    67.5,  # right_shoulder_yaw_joint
    90.0,  # left_knee_joint
    90.0,  # right_knee_joint
    22.5,  # left_ankle_pitch_joint
    22.5,  # right_ankle_pitch_joint
    22.5,  # left_ankle_roll_joint
    22.5,  # right_ankle_roll_joint
    45.0,  # left_elbow_pitch_joint
    45.0,  # right_elbow_pitch_joint
    45.0,  # left_elbow_roll_joint
    45.0,  # right_elbow_roll_joint
    ]

    # reward scale parameters
    alive_reward_scale: float = 1.0
    base_height_reward_scale: float = -10.0
    action_rate_reward_scale: float = -0.1
    tracking_lin_vel_reward_scale: float = 0.5
    tracking_ang_vel_reward_scale: float = 0.5
    orientation_reward_scale: float = -1.0
    default_pose_reward_scale: float = 5

    # reward function parameters
    base_height_target = 0.78
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)    
    termination_height: float = 0.5


    
    # commands
    num_commands = 4
    heading_command = False
    resampling_time = 8.
    
    lin_vel_x = [-1.0, 1.0] # min max [m/s]
    lin_vel_y = [-1.0, 1.0]   # min max [m/s]
    ang_vel_yaw = [-1, 1]    # min max [rad/s]
    heading = [-3.14, 3.14]


    # termination
    early_termination: bool = True
