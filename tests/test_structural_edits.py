from wormwing.connectome.loader import load_connectome
from wormwing.connectome.types import StructuralEdit, StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN


def test_apply_remove_and_retarget_chem_edit():
    conn = load_connectome("data/mock_connectome")
    c = ConnectomeCTRNN(conn)

    g = StructuralGenome(
        edits=[
            StructuralEdit(op="add_chem", src=0, dst=1, value=0.5),
            StructuralEdit(op="del_chem", src=0, dst=1),
            StructuralEdit(op="add_chem", src=2, dst=3, value=1.0),
            StructuralEdit(op="retarget_chem", src=2, dst=3, aux_src=4, aux_dst=5, value=-0.5),
        ],
        max_edits=8,
        seed=0,
    )

    c.apply_structural_genome(g)
    assert c.w_chem[0, 1] == 0.0
    assert c.w_chem[2, 3] == 0.0
    assert c.w_chem[4, 5] == -0.5
