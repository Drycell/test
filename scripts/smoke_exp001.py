from pathlib import Path

from wormwing.connectome.loader import load_connectome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv
from wormwing.evolution.structure_only import run_structure_only

conn = load_connectome("data/mock_connectome")
controller = ConnectomeCTRNN(conn)
env = WingedWorm3DEnv()
metrics = run_structure_only(controller, env, Path("runs/smoke"), generations=2, population_size=4, max_edits=4, seed=123)
print("smoke_exp001_ok", metrics)
