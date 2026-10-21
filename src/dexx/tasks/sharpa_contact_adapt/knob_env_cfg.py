"""A small, instrumented fixed-arm task, not a demo-tracking/VLA environment."""
import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import (
    FrankaSharpaEnvCfg, update_cfg_for_hand_side,
)
from .knob_spec import OBS_DIM, KnobGeometry

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets/tasks/knob"


@configclass
class SharpaKnobEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 4
    episode_length_s = 8.0
    action_space = 22
    observation_space = OBS_DIM
    state_space = 0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1/120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=2.0, replicate_physics=False)
    action_delta_rad = 0.02  # increment per 30-Hz policy step, not a torque
    goal_angle_rad = math.pi
    angle_tolerance_rad = math.radians(5)
    speed_tolerance_rad_s = 0.1
    hold_steps = 15
    trials_per_condition = 3
    friction_range = (0.4, 1.0)
    brake_range_nm = (0.002, 0.012)
    damping_range_nms = (0.0001, 0.0006)
    record_transitions = False
    robot_cfg: ArticulationCfg | None = None
    table_cfg = None
    knob_cfg: ArticulationCfg | None = None
    pregrasp = None
    geometry = None

    def __post_init__(self):
        for filename in ("knob.usda", "geometry.json", "pregrasp.json"):
            if not (ASSET_DIR / filename).is_file():
                raise FileNotFoundError(f"Missing {filename}; run scripts/knob/build_assets.py and prepare_pose.py")
        self.geometry = KnobGeometry(**json.loads((ASSET_DIR / "geometry.json").read_text()))
        self.pregrasp = json.loads((ASSET_DIR / "pregrasp.json").read_text())
        # Copy the existing robot/actuation and table definitions; do not change them.
        source = FrankaSharpaEnvCfg()
        update_cfg_for_hand_side(source, "right")
        self.robot_cfg = source.robot_cfg.copy()
        self.robot_cfg.init_state.joint_pos = self.pregrasp["joint_positions_rad"]
        self.table_cfg = source.table_cfg.copy()
        self.knob_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Knob",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ASSET_DIR / "knob.usda"),
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(self.pregrasp["fixture_position_m"]), joint_pos={"shaft": 0.0},
            ),
            actuators={"brake": IdealPDActuatorCfg(
                joint_names_expr=["shaft"], stiffness=0.0, damping=0.0,
                effort_limit=1.0, velocity_limit=100.0, armature=0.0, friction=0.0,
            )},
        )
