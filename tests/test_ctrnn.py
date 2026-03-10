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
