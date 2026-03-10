import numpy as np

from wormwing.envs.winged_worm_3d import WingedWorm3DEnv


def test_env_step_and_reward_components():
    env = WingedWorm3DEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (13,)
    obs, reward, term, trunc, info = env.step(np.array([0.1, -0.1]))
    assert np.isfinite(reward)
    for key in [
        "alive_reward",
        "forward_component",
        "height_component",
        "tilt_penalty",
        "angvel_penalty",
        "control_effort_penalty",
        "terminal_penalty",
    ]:
        assert key in info["reward_components"]
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
