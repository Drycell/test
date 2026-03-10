import numpy as np

from wormwing.connectome.loader import load_connectome


def test_mock_loader_shapes_and_gap_symmetry():
    data = load_connectome("data/mock_connectome")
    n = len(data.neuron_ids)
    assert data.chemical_weights.shape == (n, n)
    assert data.gap_weights.shape == (n, n)
    assert np.allclose(data.gap_weights, data.gap_weights.T)


def test_mask_validity():
    data = load_connectome("data/mock_connectome")
    assert np.all(np.diag(data.allowed_add_chem_mask) == 0)
    assert np.all(np.diag(data.allowed_add_gap_mask) == 0)
    for idx in data.virtual_sensor_indices:
        assert not np.any(data.allowed_add_chem_mask[:, idx])
    for idx in data.virtual_motor_indices:
        assert not np.any(data.allowed_add_chem_mask[idx, :])
