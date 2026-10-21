"""Shared SI-unit geometry and reset protocol. No Isaac/Torch imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_JOINTS = tuple("right_" + name for name in (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
))
ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class KnobGeometry:
    rotor_radius_m: float = 0.025
    rotor_height_m: float = 0.020
    bore_radius_m: float = 0.00425
    flange_radius_m: float = 0.045
    flange_height_m: float = 0.008
    pedestal_radius_m: float = 0.010
    pedestal_height_m: float = 0.035
    washer_gap_m: float = 0.001
    bolt_head_af_m: float = 0.0134
    bolt_head_depth_m: float = 0.0056
    rotor_mass_kg: float = 0.040

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Geometry fields must be finite and positive")
        if not self.bore_radius_m < self.pedestal_radius_m < self.rotor_radius_m < self.flange_radius_m:
            raise ValueError("Expected bore < pedestal < rotor < flange radii")
        if not self.bolt_head_depth_m < self.flange_height_m:
            raise ValueError("Bolt pocket must leave a roof in the flange")
        if not self.bore_radius_m < self.bolt_head_af_m / 2 < self.flange_radius_m:
            raise ValueError("Bolt pocket must be wider than bore and inside flange")

    @property
    def rotor_center_z_m(self):
        return self.flange_height_m + self.pedestal_height_m + self.washer_gap_m + self.rotor_height_m / 2

    @property
    def rotor_inertia_z(self):
        # The simulation uses a solid cylinder (shaft/bore collision omitted).
        return 0.5 * self.rotor_mass_kg * self.rotor_radius_m**2


@dataclass(frozen=True)
class PhysicalCondition:
    contact_friction: float
    brake_torque_nm: float
    damping_nms: float


class ConditionSchedule:
    """A new condition every K trials, with independent per-environment RNGs.

    force_new restarts a condition block. Repeating a trial preserves ALL z.
    Labels and z are metadata, never actor observations.
    """

    def __init__(self, num_envs, seed=42, trials_per_condition=3,
                 friction_range=(0.4, 1.0), brake_range=(0.002, 0.012),
                 damping_range=(0.0001, 0.0006)):
        if num_envs < 1 or trials_per_condition < 1:
            raise ValueError("Positive env/trial counts required")
        self.ranges = (friction_range, brake_range, damping_range)
        for lo, hi in self.ranges:
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError("Invalid physical-condition range")
        self.rngs = [random.Random(seed + 104729 * i) for i in range(num_envs)]
        self.conditions = [None] * num_envs
        self.condition_ids = [-1] * num_envs
        self.trial_ids = [-1] * num_envs
        self.trials_per_condition = trials_per_condition

    def reset_trial(self, env_id, force_new=False):
        new = force_new or self.conditions[env_id] is None or self.trial_ids[env_id] + 1 >= self.trials_per_condition
        if new:
            self.conditions[env_id] = PhysicalCondition(*[
                self.rngs[env_id].uniform(lo, hi) for lo, hi in self.ranges
            ])
            self.condition_ids[env_id] += 1
            self.trial_ids[env_id] = 0
        else:
            self.trial_ids[env_id] += 1
        return self.conditions[env_id], new


def policy_layout():
    """Explicit actor schema: measured angle is allowed, physical settings are not."""
    widths = [("q_rad", 22), ("dq_rad_s", 22), ("issued_target_rad", 22),
              ("fingertip_force_local_n", 15), ("force_valid", 5),
              ("angle_rad", 1), ("angular_velocity_rad_s", 1), ("goal_rad", 1),
              ("goal_error_rad", 1), ("trial_reset", 1)]
    offset, result = 0, {}
    for name, width in widths:
        result[name] = (offset, offset + width)
        offset += width
    return result


OBS_DIM = max(stop for _, stop in policy_layout().values())
