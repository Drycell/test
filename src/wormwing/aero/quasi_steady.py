from __future__ import annotations

import numpy as np


def wing_lift_drag(hinge_vel: float, body_vx: float, coeff_lift: float = 0.8, coeff_drag: float = 0.15) -> tuple[float, float]:
    speed = abs(hinge_vel) + abs(body_vx)
    lift = coeff_lift * hinge_vel * speed
    drag = -coeff_drag * body_vx * speed
    return float(lift), float(drag)


def wing_torque_damping(hinge_vel: float, damping: float = 0.02) -> float:
    return float(-damping * hinge_vel)
