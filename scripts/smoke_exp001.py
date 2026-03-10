from pathlib import Path

from wormwing.connectome.loader import load_connectome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import EnvConfig, WingedWorm3DEnv
from wormwing.evolution.structure_only import run_structure_only

conn = load_connectome("data/mock_connectome")
controller = ConnectomeCTRNN(conn)
env = WingedWorm3DEnv(EnvConfig(episode_seconds=2.0))
metrics = run_structure_only(
    controller,
    env,
    Path("runs/smoke"),
    generations=2,
    population_size=6,
    max_edits=4,
    seed=123,
    elite_count=2,
    mutation_cfg={"p_append": 0.35, "p_delete": 0.15, "p_replace": 0.2, "p_retarget": 0.15, "p_rescale": 0.15},
    train_eval_seeds=[0, 1],
)
print("smoke_exp001_ok", metrics)
