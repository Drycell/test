from wormwing.envs.winged_worm_3d import WingedWorm3DEnv
import numpy as np

env = WingedWorm3DEnv()
obs, _ = env.reset(seed=0)
for _ in range(20):
    obs, _, term, trunc, _ = env.step(np.array([0.0, 0.0]))
    if term or trunc:
        break
print("smoke_mujoco_ok", obs.shape)
