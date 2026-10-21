"""Batched passive brake and task criterion; Torch only, no simulator."""
import torch


def passive_brake_torque(velocity, dry_torque, damping, inertia, dt):
    """Kinetic-friction + viscous brake, capped to avoid isolated-step reversal.

    This is NOT a static-friction/breakaway solver: torque is zero at rest.
    All parameters are SI. Inertia must include any added joint armature.
    """
    if dt <= 0 or inertia <= 0:
        raise ValueError("Positive dt and inertia required")
    requested = dry_torque + damping * velocity.abs()
    stopping = inertia * velocity.abs() / dt
    return -velocity.sign() * torch.minimum(requested, stopping)


def update_hold_count(angle, velocity, goal, previous, tolerance, speed_tolerance):
    # Deliberately unwrapped: a 2*pi overshoot is not success at the original goal.
    good = ((goal - angle).abs() <= tolerance) & (velocity.abs() <= speed_tolerance)
    return torch.where(good, previous + 1, torch.zeros_like(previous))
