from pathlib import Path

from wormwing.connectome.loader import load_connectome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import EnvConfig, WingedWorm3DEnv
from wormwing.evolution.structure_only import run_structure_only


def test_evolution_short_run(tmp_path: Path):
    conn = load_connectome("data/mock_connectome")
    controller = ConnectomeCTRNN(conn)
    env = WingedWorm3DEnv(EnvConfig(episode_seconds=0.1))
    metrics = run_structure_only(controller, env, tmp_path, generations=2, population_size=3, max_edits=4, seed=0)
    assert "best_fitness" in metrics
    assert (tmp_path / "best_genome.json").exists()
