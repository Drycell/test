import numpy as np

from wormwing.connectome.loader import load_connectome
from wormwing.connectome.types import StructuralEdit, StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN


def test_ctrnn_deterministic_step_and_edits():
    conn = load_connectome("data/mock_connectome")
    c1 = ConnectomeCTRNN(conn)
    c2 = ConnectomeCTRNN(conn)
    c1.reset(seed=1)
    c2.reset(seed=1)
    obs = np.zeros(13, dtype=float)
    a1 = c1.step(obs)
    a2 = c2.step(obs)
    assert np.allclose(a1, a2)

    genome = StructuralGenome(edits=[StructuralEdit(op="add_chem", src=0, dst=1, value=1.0), StructuralEdit(op="flip_sign", src=0, dst=1)], max_edits=8, seed=1)
    c1.apply_structural_genome(genome)
    assert c1.w_chem[0, 1] == -1.0


def test_virtual_sensor_mapping_order():
    conn = load_connectome("data/mock_connectome")
    c = ConnectomeCTRNN(conn)
    c.reset(seed=0)
    obs = np.array([9.9, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 5.0, 6.0, 0, 0, 0, 0], dtype=float)
    c.step(obs)
    vals = [c.state.x[i] for i in conn.virtual_sensor_indices]
    assert np.allclose(vals, [1.0, 2.0, 3.0, 4.0, 6.0, 5.0])
