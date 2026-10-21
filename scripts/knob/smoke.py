"""Small hardware-free Isaac smoke/record run. NO training or real-robot I/O.

hold: no hand commands; must not succeed by itself.
close: scripted thumb/index/middle flexion; checks actual rotor contact, not task success.
brake: isolated shaft spin-down diagnostic, NOT a manipulation policy/example.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["hold", "close", "brake"], default="hold")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        raise SystemExit("Isaac Lab 2.2.1 / Sim 4.5 unavailable in this interpreter. See tutorial/00_setup; no simulation ran.") from e
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("--steps and --num_envs must be positive")
    for p in (args.result, args.record):
        if p is not None and p.exists():
            parser.error(f"Refusing to overwrite {p}")
    if args.mode == "brake" and args.record:
        parser.error("Shaft-drive diagnostics must not be recorded as robot interaction data")
    app = AppLauncher(args).app
    env, writer = None, None
    try:
        import torch
        import gymnasium as gym
        from dexx.tasks.sharpa_contact_adapt import register_tasks
        from dexx.tasks.sharpa_contact_adapt.knob_env_cfg import SharpaKnobEnvCfg
        from dexx.tasks.sharpa_contact_adapt.knob_spec import HAND_JOINTS, policy_layout
        from dexx.tasks.sharpa_contact_adapt.recording import TransitionWriter

        register_tasks()
        cfg = SharpaKnobEnvCfg()
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.episode_length_s = max(cfg.episode_length_s, (args.steps+10)*cfg.sim.dt*cfg.decimation)
        cfg.record_transitions = args.record is not None
        cfg.viewer.eye = (1.0, -0.8, 0.85)
        cfg.viewer.lookat = (0.48, -0.15, 0.47)
        env = gym.make("sharpa-knob-adapt-v0", cfg=cfg).unwrapped
        obs, _ = env.reset(seed=args.seed)
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            writer = TransitionWriter(args.record, {"schema": "sharpa_knob_transition_v1",
                "mode": args.mode, "seed": args.seed, "joint_order": HAND_JOINTS,
                "actor_layout": policy_layout(), "physical_settings_are_metadata_only": True})
        if args.mode == "brake":
            q = env.knob.data.joint_pos.clone()
            dq = torch.full_like(q, 5.0)
            env.knob.write_joint_state_to_sim(q, dq)
        max_force, max_drift, max_angle, successes = 0.0, 0.0, 0.0, 0
        previous_speed = 5.0
        initial = env.robot.data.default_joint_pos.clone()
        for step in range(args.steps):
            actions = torch.zeros(env.num_envs, 22, device=env.device)
            if args.mode == "close" and step < 60:
                for suffix in ("thumb_MCP_FE", "index_MCP_FE", "middle_MCP_FE"):
                    actions[:, HAND_JOINTS.index("right_"+suffix)] = 0.1
            obs, reward, terminated, truncated, extras = env.step(actions)
            if not torch.isfinite(obs["policy"]).all():
                raise RuntimeError("Non-finite observation")
            metrics = extras["task_metrics"]
            if metrics["failure"].any():
                raise RuntimeError("Task reported failure")
            successes += int(metrics["success"].sum())
            force, _ = env._forces()
            max_force = max(max_force, float(force.norm(dim=-1).max()))
            max_angle = max(max_angle, float(env.knob.data.joint_pos.abs().max()))
            max_drift = max(max_drift, float((env.robot.data.joint_pos[:, env.arm_ids]-initial[:, env.arm_ids]).abs().max()))
            limits = env.robot.data.soft_joint_pos_limits
            q = env.robot.data.joint_pos
            if ((q < limits[..., 0]-.05) | (q > limits[..., 1]+.05)).any():
                raise RuntimeError("Robot crossed joint limits")
            if args.mode == "brake":
                speed = float(env.knob.data.joint_vel.abs().max())
                if max_force > .01 or speed > previous_speed + .02:
                    raise RuntimeError("Brake diagnostic is not isolated/passive")
                previous_speed = speed
            if writer:
                writer.write_batch(extras["transition"])
        if args.mode == "hold" and (successes or max_angle > .05):
            raise RuntimeError("Zero-action rotor drift/success; inspect initial collisions")
        if args.mode == "close" and max_force < .05:
            raise RuntimeError("No verified fingertip-to-rotor contact; inspect pregrasp")
        if args.mode == "brake" and previous_speed > .1:
            raise RuntimeError("Shaft did not stop within diagnostic budget")
        if max_drift > .1:
            raise RuntimeError("Arm hold drift too large; inspect holding controller/geometry")
        # Confirm subset trial reset retains physics and new-condition reset changes ID.
        env.restart_trial(torch.tensor([0], device=env.device), new_condition=True)
        old_condition = env.schedule.conditions[0]
        old_id = env.schedule.condition_ids[0]
        env.restart_trial(torch.tensor([0], device=env.device))
        if env.schedule.conditions[0] != old_condition or env.schedule.condition_ids[0] != old_id:
            raise RuntimeError("Trial reset unexpectedly changed physical condition")
        env.restart_trial(torch.tensor([0], device=env.device), new_condition=True)
        if env.schedule.condition_ids[0] != old_id+1:
            raise RuntimeError("New-condition reset did not advance ID")
        result = {"ok": True, "mode": args.mode, "steps": args.steps,
                  "max_contact_force_n": max_force, "max_arm_drift_rad": max_drift,
                  "max_rotor_angle_rad": max_angle, "successes": successes,
                  "note": "Engineering smoke only; not a learned-policy success rate."}
        args.result.parent.mkdir(parents=True, exist_ok=True)
        with args.result.open("x") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result))
    finally:
        if writer:
            writer.close()
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
