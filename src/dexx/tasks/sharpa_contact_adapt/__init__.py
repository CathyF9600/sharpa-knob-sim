"""Standalone contact-adaptation tasks; importing offline helpers needs no simulator.

Call register_tasks() after starting Isaac Sim to register the Gym environment.
"""


def register_tasks():
    import gymnasium as gym

    task_id = "sharpa-knob-adapt-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="dexx.tasks.sharpa_contact_adapt.knob_env:SharpaKnobEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "dexx.tasks.sharpa_contact_adapt.knob_env_cfg:SharpaKnobEnvCfg",
            },
        )
