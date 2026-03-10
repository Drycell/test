import numpy as np

from wormwing.connectome.loader import load_connectome


def test_mock_loader_shapes_and_gap_symmetry():
    data = load_connectome("data/mock_connectome")
    n = len(data.neuron_ids)
    assert data.chemical_weights.shape == (n, n)
    assert data.gap_weights.shape == (n, n)
    assert np.allclose(data.gap_weights, data.gap_weights.T)
