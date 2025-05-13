# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .leatherback import LEATHERBACK_CFG
from .waypoint import WAYPOINT_CFG
from .cone import CONES_CFG
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.assets import RigidObjectCollectionCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4          # Decimation - number of time steps between actions, it was 2
    episode_length_s = 20.0 # Max each episode should last in seconds, 30 s seems a lot
    action_space = 2        # Number of actions the neural network shuold return   
    observation_space = 8   # Number of observations fed into neural network
    state_space = 0         # Observations to be used in Actor Critic Training
    num_goals = 10
    env_spacing = 32.0 # depends on the ammount of Goals, 32 is a lot
    course_length_coefficient = 2.5
    course_width_coefficient = 2.0
    # simulation frames Hz
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Waypoints
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG
    # Spawning Traffic Cones
    cone_collection_cfg: RigidObjectCollectionCfg = CONES_CFG  # Ensure naming consistency
    
    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    # scene - 4096 environments
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)