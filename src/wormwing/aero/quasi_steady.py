from __future__ import annotations

import math


def wing_lift_drag(
    hinge_vel: float,
    body_vx: float,
    wing_span_m: float,
    wing_chord_m: float,
    air_density: float = 1.225,
    cl: float = 1.0,
    cd: float = 1.4,
) -> tuple[float, float]:
    """Simple deterministic quasi-steady aerodynamics for micro-scale wing actuation.

    Returns lift (z) and drag (x) force magnitudes in Newton.
    """
    wing_tip_speed = abs(hinge_vel) * max(wing_span_m, 1e-6)
    rel_speed = max(1e-5, abs(body_vx) + wing_tip_speed)
    area = max(1e-10, wing_span_m * wing_chord_m)
    q = 0.5 * air_density * rel_speed * rel_speed
    lift = q * cl * area * (1.0 if hinge_vel >= 0.0 else -1.0)
    drag = q * cd * area * (-1.0 if body_vx >= 0.0 else 1.0)
    return float(lift), float(drag)


def wing_torque_damping(hinge_vel: float, damping: float = 2e-10) -> float:
    """Viscous damping around wing hinge for numerical stability."""
    return float(-damping * hinge_vel)


def body_drag(body_vx: float, area_m2: float, air_density: float = 1.225, cd: float = 0.9) -> float:
    speed = abs(body_vx)
    if speed < 1e-7:
        return 0.0
    force = 0.5 * air_density * cd * area_m2 * speed * speed
    return float(-math.copysign(force, body_vx))
