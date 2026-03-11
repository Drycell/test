from wormwing.connectome.loader import load_connectome


def test_real_connectome_bundle_loads():
    data = load_connectome("data/real_connectome")
    assert len(data.neuron_ids) > 300
    assert data.chemical_weights.shape[0] == data.gap_weights.shape[0]
    assert len(data.sensor_node_indices) > 0
    assert len(data.motor_node_indices) > 0
