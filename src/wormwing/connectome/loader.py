from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wormwing.connectome.types import ConnectomeData

VIRTUAL_SENSORS = ["VS_ROLL", "VS_PITCH", "VS_DROLL", "VS_DPITCH", "VS_ALT", "VS_DALT"]
VIRTUAL_MOTORS = ["VM_WING_L", "VM_WING_R"]


def _clean_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(" ", "", regex=False)


def load_connectome(data_dir: str | Path, append_virtual_nodes: bool = True) -> ConnectomeData:
    data_path = Path(data_dir)
    neurons = pd.read_csv(data_path / "neurons.csv")
    chem = pd.read_csv(data_path / "chemical_synapses.csv")
    gap = pd.read_csv(data_path / "gap_junctions.csv")

    neurons["neuron_id"] = _clean_id_series(neurons["neuron_id"])
    chem["src"] = _clean_id_series(chem["src"])
    chem["dst"] = _clean_id_series(chem["dst"])
    gap["a"] = _clean_id_series(gap["a"])
    gap["b"] = _clean_id_series(gap["b"])

    neuron_ids = neurons["neuron_id"].tolist()
    if append_virtual_nodes:
        neuron_ids = neuron_ids + VIRTUAL_SENSORS + VIRTUAL_MOTORS

    index = {n: i for i, n in enumerate(neuron_ids)}
    n = len(neuron_ids)
    chem_w = np.zeros((n, n), dtype=np.float64)
    gap_w = np.zeros((n, n), dtype=np.float64)

    sign_missing = "sign" not in chem.columns
    for _, row in chem.iterrows():
        if row["src"] not in index or row["dst"] not in index:
            continue
        s, d = index[row["src"]], index[row["dst"]]
        sign = float(row.get("sign", 1.0))
        chem_w[s, d] += float(row["weight"]) * sign

    for _, row in gap.iterrows():
        if row["a"] not in index or row["b"] not in index:
            continue
        a, b = index[row["a"]], index[row["b"]]
        w = float(row["weight"])
        gap_w[a, b] += w
        gap_w[b, a] += w

    sensor_idxs = [i for i, x in enumerate(neurons["is_sensor"].tolist()) if int(x) == 1]
    motor_idxs = [i for i, x in enumerate(neurons["is_motor"].tolist()) if int(x) == 1]

    virtual_sensor_indices = [index[k] for k in VIRTUAL_SENSORS] if append_virtual_nodes else []
    virtual_motor_indices = [index[k] for k in VIRTUAL_MOTORS] if append_virtual_nodes else []

    add_chem = np.ones_like(chem_w, dtype=bool)
    np.fill_diagonal(add_chem, False)
    del_chem = chem_w != 0.0
    add_gap = np.ones_like(gap_w, dtype=bool)
    np.fill_diagonal(add_gap, False)
    del_gap = gap_w != 0.0

    if append_virtual_nodes:
        add_chem[:, virtual_sensor_indices] = False
        add_chem[virtual_motor_indices, :] = False

    return ConnectomeData(
        neuron_ids=neuron_ids,
        chemical_weights=chem_w,
        gap_weights=gap_w,
        sensor_node_indices=sensor_idxs,
        motor_node_indices=motor_idxs,
        metadata={"sign_assumed_positive": sign_missing, "source_dir": str(data_path)},
        allowed_add_chem_mask=add_chem,
        allowed_del_chem_mask=del_chem,
        allowed_add_gap_mask=add_gap,
        allowed_del_gap_mask=del_gap,
        virtual_sensor_indices=virtual_sensor_indices,
        virtual_motor_indices=virtual_motor_indices,
    )
