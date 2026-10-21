"""Passive shaft, measured goal, independently scheduled contact conditions.

The arm holds a pregrasp pose. Policy controls 22 named Sharpa joints only.
Terminal transitions are captured BEFORE DirectRLEnv's automatic reset.
"""
from dataclasses import asdict
import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply_inverse

from .knob_dynamics import passive_brake_torque, update_hold_count
from .knob_spec import ARM_JOINTS, FINGERS, HAND_JOINTS, OBS_DIM, ConditionSchedule


class SharpaKnobEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if not math.isfinite(cfg.goal_angle_rad) or abs(cfg.goal_angle_rad) <= 2*cfg.angle_tolerance_rad:
            raise ValueError("Goal must require non-trivial rotation from zero")
        if cfg.hold_steps < 1 or cfg.action_delta_rad <= 0:
            raise ValueError("Positive hold_steps and action_delta_rad required")
        super().__init__(cfg, render_mode, **kwargs)
        self.hand_ids, names = self.robot.find_joints(list(HAND_JOINTS), preserve_order=True)
        if tuple(names) != HAND_JOINTS:
            raise RuntimeError(f"Hand joint mapping mismatch: {names}")
        self.arm_ids, names = self.robot.find_joints(list(ARM_JOINTS), preserve_order=True)
        if tuple(names) != ARM_JOINTS:
            raise RuntimeError(f"Arm joint mapping mismatch: {names}")
        self.tip_ids, _ = self.robot.find_bodies([f"right_{f}_elastomer" for f in FINGERS], preserve_order=True)
        self.shaft_ids, _ = self.knob.find_joints("shaft")
        if len(self.tip_ids) != 5 or len(self.shaft_ids) != 1:
            raise RuntimeError("Expected five fingertips and one shaft")
        self.schedule = ConditionSchedule(self.num_envs, cfg.seed, cfg.trials_per_condition,
                                          cfg.friction_range, cfg.brake_range_nm, cfg.damping_range_nms)
        self.goal = torch.full((self.num_envs,), cfg.goal_angle_rad, device=self.device)
        self.target = self.robot.data.default_joint_pos.clone()
        self.actions = torch.zeros(self.num_envs, 22, device=self.device)
        self.brake = torch.zeros(self.num_envs, device=self.device)
        self.damping = torch.zeros_like(self.brake)
        self.hold = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.fresh = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_flag = torch.ones_like(self.fresh)
        self.reward = torch.zeros_like(self.brake)
        self.force_new = False
        self.before = None
        # Explicit baseline hand material. Rotor friction varies independently.
        view = self.robot.root_physx_view
        material = view.get_material_properties().clone()
        material[..., :2] = 0.8
        material[..., 2] = 0.0
        view.set_material_properties(material, torch.arange(self.num_envs, device="cpu"))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.knob = Articulation(self.cfg.knob_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.contact_sensors = []
        for finger in FINGERS:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/right_{finger}_elastomer",
                filter_prim_paths_expr=["/World/envs/env_.*/Knob/rotor"],
                history_length=3, update_period=0.0,
            ))
            self.contact_sensors.append(sensor)
            self.scene.sensors[finger] = sensor
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["knob"] = self.knob
        self.scene.rigid_objects["table"] = self.table
        light = sim_utils.DomeLightCfg(intensity=1800.0)
        light.func("/World/Light", light)

    def _forces(self):
        forces = []
        for sensor in self.contact_sensors:
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                raise RuntimeError("Filtered fingertip-to-rotor force unavailable; check sensor body path")
            forces.append(matrix.sum(dim=(1, 2)))
        world = torch.stack(forces, dim=1)
        valid = torch.isfinite(world).all(dim=-1) & self.fresh[:, None]
        world = torch.nan_to_num(world) * valid[..., None]
        quats = self.robot.data.body_link_state_w[:, self.tip_ids, 3:7]
        local = quat_apply_inverse(quats.reshape(-1, 4), world.reshape(-1, 3)).reshape(-1, 5, 3)
        # ContactSensor exposes resultant normal-contact vectors, not a shear map.
        return local, valid

    def _snapshot(self):
        force, valid = self._forces()
        return {
            "q_rad": self.robot.data.joint_pos[:, self.hand_ids].clone(),
            "dq_rad_s": self.robot.data.joint_vel[:, self.hand_ids].clone(),
            "issued_target_rad": self.target[:, self.hand_ids].clone(),
            "force_local_n": force.clone(), "force_valid": valid.clone(),
            "angle_rad": self.knob.data.joint_pos[:, self.shaft_ids[0]].clone(),
            "angular_velocity_rad_s": self.knob.data.joint_vel[:, self.shaft_ids[0]].clone(),
            "goal_rad": self.goal.clone(), "trial_reset": self.reset_flag.clone(),
        }

    def _get_observations(self):
        s = self._snapshot()
        obs = torch.cat([s["q_rad"], s["dq_rad_s"], s["issued_target_rad"],
                         s["force_local_n"].flatten(1), s["force_valid"].float(),
                         s["angle_rad"][:, None], s["angular_velocity_rad_s"][:, None],
                         s["goal_rad"][:, None], (s["goal_rad"]-s["angle_rad"])[:, None],
                         s["trial_reset"].float()[:, None]], dim=1)
        if obs.shape[1] != OBS_DIM:
            raise RuntimeError(f"Unexpected observation width {obs.shape}")
        return {"policy": obs}

    def _pre_physics_step(self, actions):
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite [num_envs,22] actions")
        self.before = self._snapshot()
        self.proposed_actions = actions.clone()
        self.actions = actions.clamp(-1, 1).clone()
        limits = self.robot.data.soft_joint_pos_limits[:, self.hand_ids]
        proposal = self.target[:, self.hand_ids] + self.cfg.action_delta_rad * self.actions
        self.target[:, self.hand_ids] = proposal.clamp(limits[..., 0], limits[..., 1])
        self.reset_flag[:] = False
        self.fresh[:] = True  # consumed only after this policy step's physics updates

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target)
        velocity = self.knob.data.joint_vel[:, self.shaft_ids[0]]
        torque = passive_brake_torque(velocity, self.brake, self.damping,
                                      self.cfg.geometry.rotor_inertia_z, self.physics_dt)
        self.knob.set_joint_effort_target(torque[:, None], joint_ids=self.shaft_ids)

    def _get_dones(self):
        after = self._snapshot()
        angle, speed = after["angle_rad"], after["angular_velocity_rad_s"]
        self.hold = update_hold_count(angle, speed, self.goal, self.hold,
                                      self.cfg.angle_tolerance_rad, self.cfg.speed_tolerance_rad_s)
        finite = (torch.isfinite(angle) & torch.isfinite(speed) &
                  torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
        success = (self.hold >= self.cfg.hold_steps) & finite
        failure = ~finite | (angle.abs() > 2*math.pi)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        error = (self.goal-angle).abs()
        prior_error = (self.goal-self.before["angle_rad"]).abs()
        self.reward = torch.nan_to_num(5*(prior_error-error) + torch.exp(-3*error)
                                       - .002*self.actions.square().sum(-1)
                                       + 5*success.float() - 5*failure.float(),
                                       nan=-5.0, posinf=-5.0, neginf=-5.0)
        self.extras["task_metrics"] = {
            "success": success.clone(), "failure": failure.clone(),
            "angle_error_rad": error.clone(), "hold_steps": self.hold.clone(),
        }
        if self.cfg.record_transitions:
            self.extras["transition"] = {
                "before": self.before, "after": after,
                "proposed_action": self.proposed_actions.clone(),
                "issued_target_rad": after["issued_target_rad"].clone(),
                "duration_s": self.step_dt,
                "terminated": (success | failure).clone(), "truncated": timeout.clone(),
                "condition_id": list(self.schedule.condition_ids),
                "trial_id": list(self.schedule.trial_ids),
                "physical_settings_metadata_only": [asdict(c) for c in self.schedule.conditions],
            }
        return success | failure, timeout

    def _get_rewards(self):
        return self.reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        changed = []
        view = self.knob.root_physx_view
        materials = view.get_material_properties().clone()
        for i in env_ids.tolist():
            condition, is_new = self.schedule.reset_trial(i, force_new=self.force_new)
            if is_new:
                materials[i, :, :2] = condition.contact_friction
                materials[i, :, 2] = 0.0
                changed.append(i)
            self.brake[i] = condition.brake_torque_nm
            self.damping[i] = condition.damping_nms
        if changed:
            view.set_material_properties(materials, torch.tensor(changed, device="cpu"))
        for asset in (self.robot, self.knob):
            root = asset.data.default_root_state[env_ids].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
            asset.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
            q = asset.data.default_joint_pos[env_ids].clone()
            asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.target[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.robot.set_joint_position_target(self.target[env_ids], env_ids=env_ids)
        self.knob.set_joint_effort_target(torch.zeros(len(env_ids), 1, device=self.device), env_ids=env_ids)
        self.actions[env_ids] = 0
        self.hold[env_ids] = 0
        self.reset_flag[env_ids] = True
        self.fresh[env_ids] = False
        for sensor in self.contact_sensors:
            sensor.reset(env_ids)

    def reset(self, *, seed=None, options=None):
        # A full public reset starts a fresh condition block. Auto resets continue
        # trials; use restart_trial() for a manual matched-condition query reset.
        if seed is not None:
            self.schedule = ConditionSchedule(self.num_envs, seed, self.cfg.trials_per_condition,
                                              self.cfg.friction_range, self.cfg.brake_range_nm,
                                              self.cfg.damping_range_nms)
        self.force_new = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.force_new = False

    def restart_trial(self, env_ids=None, *, new_condition=False):
        """Reset simulator state, not caller-owned context. Clear context iff z changes."""
        self.force_new = new_condition
        try:
            self._reset_idx(env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            return self._get_observations()
        finally:
            self.force_new = False
