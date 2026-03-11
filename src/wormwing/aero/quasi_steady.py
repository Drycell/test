from __future__ import annotations

import math


def wing_lift_drag(
    hinge_vel: float,
    body_vx: float,
    wing_span_m: float,
    wing_chord_m: float,
    air_density: float = 1.225,
    hinge_angle: float = 0.0,
    beat_freq_hz: float = 400.0,
    cl: float = 0.25,
    cd: float = 1.0,
    thrust_ratio: float = 0.03,
) -> tuple[float, float]:
    """Quasi-steady low-Re estimate for tiny flapping wings.

    Uses beat-frequency-based tip speed so lift remains non-zero even when
    position actuators hold hinge velocity near zero.
    """
    area = max(1e-12, wing_span_m * wing_chord_m)
    flap_speed = 2.0 * math.pi * max(1.0, beat_freq_hz) * max(1e-6, wing_span_m) * abs(math.sin(float(hinge_angle)))
    rel_speed = max(1e-5, flap_speed + 0.1 * abs(hinge_vel) * wing_span_m)
    q = 0.5 * air_density * rel_speed * rel_speed
    lift = q * cl * area
    drag = q * cd * area
    thrust = thrust_ratio * lift
    signed_drag = -math.copysign(drag, body_vx if abs(body_vx) > 1e-8 else 1.0)
    return float(lift), float(signed_drag + thrust)


def wing_torque_damping(hinge_vel: float, damping: float = 2e-10) -> float:
    return float(-damping * hinge_vel)


def body_drag(body_vx: float, area_m2: float, air_density: float = 1.225, cd: float = 0.9) -> float:
    speed = abs(body_vx)
    if speed < 1e-8:
        return 0.0
    force = 0.5 * air_density * cd * area_m2 * speed * speed
    return float(-math.copysign(force, body_vx))
