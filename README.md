# wormwing

Structure-first evolution research scaffold for a connectome-derived CTRNN controlling a 3D winged worm body.

## Quickstart

```bash
make install
make smoke
make smoke-exp001
```

## Main experiment

```bash
make train-exp001
make eval-latest
make report-latest
```

## Connectome modes

- Mock mode (default): `data/mock_connectome`
- Real bundle mode: `data/real_connectome` (derived from OpenWorm c302 hermaphrodite edgelist)

Use `configs/experiment/exp001_real.yaml` to run with the real bundle.

## Korean Guide

- See `GUIDE_KO.md` for Korean setup and run instructions.
