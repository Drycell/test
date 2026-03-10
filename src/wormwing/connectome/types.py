from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class ConnectomeData:
    neuron_ids: list[str]
    chemical_weights: np.ndarray
    gap_weights: np.ndarray
    sensor_node_indices: list[int]
    motor_node_indices: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_add_chem_mask: np.ndarray | None = None
    allowed_del_chem_mask: np.ndarray | None = None
    allowed_add_gap_mask: np.ndarray | None = None
    allowed_del_gap_mask: np.ndarray | None = None
    virtual_sensor_indices: list[int] = field(default_factory=list)
    virtual_motor_indices: list[int] = field(default_factory=list)


@dataclass
class StructuralEdit:
    op: Literal["add_chem", "del_chem", "add_gap", "del_gap", "flip_sign", "retarget_chem"]
    src: int
    dst: int
    aux_src: int | None = None
    aux_dst: int | None = None
    value: float | None = None
    meta: dict[str, Any] | None = None


@dataclass
class StructuralGenome:
    edits: list[StructuralEdit]
    max_edits: int
    seed: int


@dataclass
class HybridGenome:
    structural: StructuralGenome
    bias_delta: np.ndarray | None = None
    tau_log_scale_delta: np.ndarray | None = None
    edited_edge_scale_delta: np.ndarray | None = None
