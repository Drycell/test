# AGENTS.md — Structure-First Evolution of a C. elegans Connectome for Novel-Organ Control

## 0. Purpose and execution contract
This file is the **single source of truth** for Codex when implementing this repository.
The repository must be good enough to run **one complete research experiment** end-to-end, not just a demo.

The canonical v1 research experiment is:

> **EXP-001 — Minimal structural evolution of a connectome-derived controller for 3D wing-assisted stabilization.**

Codex must follow these rules:
- Treat this file as binding unless the user explicitly overrides it.
- Build the smallest system that can run EXP-001 end-to-end.
- Prefer a clear, testable implementation over a biologically maximal one.
- Keep the codebase runnable from a clean checkout at all times.
- Work in small milestones and run the relevant validation commands after each milestone.
- Do not add network-dependent runtime behavior.
- Do not replace the connectome controller with a generic RNN or MLP. Baselines may exist, but the connectome controller is the main system.
- Structural evolution is the **primary experimental variable**. Parameter tuning is secondary.
- Keep file names, code, comments, tests, and docs in English.
- If real connectome data is unavailable, fall back to bundled mock data. Missing real data must never block local runs.

---

## 1. Mission
Build a reproducible 3D research prototype in which a **C. elegans connectome-derived recurrent controller** controls a **worm-like body with a novel bilateral wing organ system**, and the main question is answered by **structural evolution of the connectome graph**.

The repository must support:
1. loading a real connectome from local CSV files when available,
2. running fully offline with a bundled mock connectome,
3. simulating a 3D winged worm body in MuJoCo,
4. evolving structural edits to the connectome,
5. comparing structural evolution against simpler baselines,
6. exporting reproducible metrics, checkpoints, graph diffs, and replay artifacts.

This repository is a **research platform for one focused study**, not a general-purpose AI agent framework and not a biology-perfect reconstruction.

---

## 2. Canonical research framing
The research question for v1 is:

> **What minimal structural edits to a connectome-derived controller enable stable control of a worm-like body with a new wing organ system in 3D?**

The canonical claim the repository should make testable is:

> Starting from a fixed connectome prior, bounded structural edits can produce better wing-assisted stabilization and glide-like control than fixed-connectome readout-only control, and the learned edits can be analyzed as graph-level adaptations.

Secondary question:

> Does a structure-first hybrid approach (structure first, local parameter refinement second) outperform structure-only evolution under the same graph edit budget?

### 2.1 Hypotheses
Codex should keep the implementation aligned with these hypotheses:
- **H1**: A fixed connectome with only a simple output readout is insufficient for robust novel-organ control.
- **H2**: Structural evolution alone can create useful control pathways from new sensory inputs to new motor outputs.
- **H3**: Successful solutions will concentrate edits in a small subset of sensory-to-interneuron-to-motor routes rather than requiring uniform global rewiring.
- **H4**: A structure-first hybrid stage will improve performance and robustness after a good structural scaffold is discovered.

### 2.2 v1 success criteria
EXP-001 is considered successful if all of the following are true:
- The repository can run end-to-end from a clean checkout with the bundled mock connectome.
- A fixed-connectome baseline runs and logs metrics.
- A structure-only evolutionary run completes and produces non-empty structural edits.
- At least one evolved controller can complete the full evaluation episode duration without immediate crash in a majority of evaluation seeds.
- The run artifacts include the best edit list, before/after graph exports, metrics, and a replayable episode artifact.
- The analysis scripts can produce at least:
  - learning curve,
  - best-vs-baseline comparison,
  - edit-count histogram,
  - graph edit summary.

A stronger success condition, if achieved, is that structure-only or structure-first hybrid clearly outperforms readout-only on the main evaluation metric.

---

## 3. Hard decisions frozen for v1
These are not open design questions.

### 3.1 Physics engine
- Use **MuJoCo** with the official Python package.
- The environment must follow **Gymnasium-style** `reset()` / `step()` semantics.
- Headless Linux runs are first-class. Viewer-based debugging is optional.

### 3.2 Body plan
- One free-floating torso body.
- Torso may be represented as:
  - one capsule, or
  - a short chain of 2 capsules if needed for stability.
- One left wing and one right wing.
- Exactly **one hinge joint per wing** in v1.
- Exactly **2 continuous motor outputs** in v1: left and right wing drive.

### 3.3 Aerodynamics
- Use a **simple deterministic quasi-steady force model** implemented in Python.
- Do not use CFD.
- Do not implement MuJoCo plugins in v1.
- Prioritize stable, explainable, deterministic behavior over realism.

### 3.4 Controller family
- The primary controller is a **CTRNN** derived from connectome structure.
- Chemical synapses map to directed weighted interactions.
- Gap junctions map to symmetric coupling.
- The controller state must be explicit and serializable.
- The controller must support extra virtual nodes that represent the new organ interface.

### 3.5 Evolution strategy
Use the following staged sequence:
- **Stage 0**: hand-coded / PD smoke baseline.
- **Stage 1**: fixed-connectome baseline with simple readout.
- **Stage 2**: **structure-only evolution** (primary experiment).
- **Stage 3**: structure-first hybrid refinement (optional but targeted for v1 if time allows).

### 3.6 Scope restriction
This v1 is about **one experiment**, not many tasks.
The canonical task is **3D wing-assisted stabilization with forward glide reward shaping**.

---

## 4. Non-goals
The following are explicitly out of scope for v1:
- language modeling,
- NLP or toy-language experiments,
- full spiking neuron simulation,
- detailed biomechanics of real C. elegans,
- full worm muscle reconstruction,
- high-fidelity aerodynamics,
- co-evolution of body morphology,
- distributed cluster schedulers,
- cloud services,
- web dashboards,
- mandatory external downloads at runtime,
- generic RL infrastructure that hides the connectome details.

---

## 5. Canonical experiment definition: EXP-001

## 5.1 Experiment title
**EXP-001 — Minimal structural evolution for 3D wing-assisted stabilization and controlled glide**

## 5.2 Biological/computational idea
Start from a base connectome graph `G0`, add a new organ interface via virtual nodes, and evolve **graph edits** that let the controller use the new sensory/motor channels to stabilize and guide the body.

The new organ system is represented by:
- virtual sensory nodes carrying body state relevant to wing control,
- virtual motor nodes driving left and right wing hinges.

The body is not required to hover. It only needs to avoid rapid destabilization, reduce crash rate, and achieve controlled wing-assisted glide-like motion.

## 5.3 Canonical task definition
Initial conditions:
- body starts above the ground at moderate height,
- a small forward velocity is injected at reset,
- small random pose perturbation is applied.

Objective:
- maintain an upright-ish body orientation,
- avoid crashing,
- stay aloft as long as possible,
- preserve or improve forward travel.

Termination conditions:
- torso height below crash threshold,
- excessive tilt,
- NaN / numerical instability,
- timeout.

## 5.4 Canonical reward structure
Use a dense shaped reward. Default implementation should follow this structure:

```text
reward =
    + alive_bonus
    + forward_progress_bonus
    + height_bonus
    - tilt_penalty
    - angular_velocity_penalty
    - control_effort_penalty
    - structural_complexity_penalty(optional during evolution objective)
    + terminal_bonus_or_penalty
```

Default behavior:
- survival and stabilization dominate early learning,
- forward progress is helpful but smaller than not crashing,
- control effort is regularized,
- structural edit cost is not applied per step, but at episode aggregation time or fitness time.

Codex may tune numeric constants conservatively, but the reward decomposition must remain inspectable and logged by component.

## 5.5 Canonical primary metrics
The main reported metrics for EXP-001 are:
- `success_rate`: fraction of evaluation episodes that survive to timeout,
- `mean_total_reward`,
- `mean_episode_length`,
- `mean_forward_distance`,
- `mean_height`,
- `mean_tilt_error`,
- `mean_control_effort`,
- `edit_count`,
- `graph_edit_distance_proxy`,
- `robust_success_rate` under perturbations.

The primary scientific comparison is:
- fixed-connectome baseline vs structure-only evolution.

The secondary comparison is:
- structure-only vs structure-first hybrid.

---

## 6. Target platforms and execution model

### Primary execution environment
- Ubuntu 24.04.x.
- Dockerized execution.
- Headless-friendly MuJoCo rendering setup.

### Secondary execution environment
- Apple Silicon macOS.
- Native `venv` execution for local development and viewer debugging.

### Baseline software assumptions
- Python 3.11.
- `mujoco`, `numpy`, `scipy`, `gymnasium`, `pyyaml`, `cmaes`, `networkx`, `matplotlib`, `pandas`, `pytest`, `ruff`.
- Avoid large ML frameworks unless clearly needed. Pure NumPy is preferred for v1.

### Parallel evaluation rule
- CPU correctness must work without GPU.
- Use **spawn-safe multiprocessing** so macOS also works.
- Worker functions must be top-level and pickle-safe.
- Do not require notebooks or REPL-only behavior.

---

## 7. Repository layout
Create this structure unless a very small improvement is clearly necessary.

```text
.
├── AGENTS.md
├── README.md
├── Makefile
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements/
│   ├── base.txt
│   └── dev.txt
├── configs/
│   ├── env/
│   │   └── winged_worm_3d.yaml
│   ├── controller/
│   │   └── ctrnn.yaml
│   ├── evo/
│   │   ├── fixed_readout.yaml
│   │   ├── structure_only.yaml
│   │   └── structure_first_hybrid.yaml
│   └── experiment/
│       ├── exp001_mock.yaml
│       └── exp001_real.yaml
├── data/
│   ├── mock_connectome/
│   └── README.md
├── scripts/
│   ├── smoke_mujoco.py
│   ├── smoke_exp001.py
│   ├── train.py
│   ├── eval.py
│   ├── render_episode.py
│   ├── make_report.py
│   └── prepare_connectome.py
├── src/
│   └── wormwing/
│       ├── aero/
│       ├── analysis/
│       ├── connectome/
│       ├── controllers/
│       ├── envs/
│       ├── evolution/
│       ├── experiments/
│       ├── io/
│       └── utils/
├── tests/
└── runs/   # gitignored
```

---

## 8. Core data contracts
Use explicit dataclasses or typed containers. Avoid opaque dicts where schemas matter.

### 8.1 `ConnectomeData`
Minimum fields:
- `neuron_ids: list[str]`
- `chemical_weights: np.ndarray` shape `(N, N)`
- `gap_weights: np.ndarray` shape `(N, N)`
- `sensor_node_indices: list[int]`
- `motor_node_indices: list[int]`
- `metadata: dict[str, Any]`
- `allowed_add_chem_mask: np.ndarray | None`
- `allowed_del_chem_mask: np.ndarray | None`
- `allowed_add_gap_mask: np.ndarray | None`
- `allowed_del_gap_mask: np.ndarray | None`
- `virtual_sensor_indices: list[int]`
- `virtual_motor_indices: list[int]`

### 8.2 `ControllerState`
Minimum fields:
- `x: np.ndarray` hidden state
- `t: float`

### 8.3 `StructuralEdit`
Each edit must be explicit. Minimum fields:
- `op: Literal["add_chem", "del_chem", "add_gap", "del_gap", "flip_sign", "retarget_chem"]`
- `src: int`
- `dst: int`
- `aux_src: int | None = None`
- `aux_dst: int | None = None`
- `value: float | None = None`
- `meta: dict[str, Any] | None = None`

Notes:
- `retarget_chem` means: remove one source/destination pairing and create another chemical edge.
- Use it only if needed. It may be implemented later by composing delete + add.

### 8.4 `StructuralGenome`
Minimum fields:
- `edits: list[StructuralEdit]`
- `max_edits: int`
- `seed: int`

### 8.5 `HybridGenome`
Minimum fields:
- `structural: StructuralGenome`
- `bias_delta: np.ndarray | None`
- `tau_log_scale_delta: np.ndarray | None`
- `edited_edge_scale_delta: np.ndarray | None`

### 8.6 `EpisodeMetrics`
Minimum fields:
- `total_reward`
- `episode_length`
- `forward_distance`
- `mean_height`
- `mean_tilt_error`
- `mean_angvel_penalty`
- `mean_control_effort`
- `termination_reason`
- `edit_count`
- `graph_edit_distance_proxy`

### 8.7 `RunManifest`
Minimum fields:
- resolved config,
- git commit if available,
- python version,
- platform,
- seed list,
- connectome mode,
- experiment name,
- timestamps.

---

## 9. Connectome input specification
Support two modes.

## 9.1 Mock mode (required)
Bundle a small mock connectome so the repo works offline from a clean checkout.

Mock connectome requirements:
- 16–32 real neurons,
- both directed chemical edges and symmetric gap edges,
- designated sensor-like and motor-like subsets,
- deterministic CSV files committed to the repo,
- simple enough for unit tests and short smoke evolution,
- includes clear masks for allowed edits.

Mock mode is the default for CI, smoke tests, and first-run local development.

## 9.2 Real mode (required interface, optional actual data)
Load from a local directory containing:
- `neurons.csv`
- `chemical_synapses.csv`
- `gap_junctions.csv`

Minimum CSV schema:

### `neurons.csv`
Columns:
- `neuron_id`
- `class` (optional)
- `is_sensor` (0/1)
- `is_motor` (0/1)
- `x` (optional)
- `y` (optional)
- `z` (optional)

### `chemical_synapses.csv`
Columns:
- `src`
- `dst`
- `weight`
- `sign` (optional, values `+1` or `-1`)

### `gap_junctions.csv`
Columns:
- `a`
- `b`
- `weight`

If `sign` is absent for real chemical edges, assume positive base weights and record that assumption in metadata.

## 9.3 Virtual novel-organ interface
The canonical v1 implementation must append virtual nodes to the base graph:

Virtual sensory nodes:
- `VS_ROLL`
- `VS_PITCH`
- `VS_DROLL`
- `VS_DPITCH`
- `VS_ALT`
- `VS_DALT`

Virtual motor nodes:
- `VM_WING_L`
- `VM_WING_R`

Rules:
- virtual sensory nodes receive clamped normalized observations each control step,
- virtual sensory nodes do not receive incoming graph edges,
- virtual motor nodes produce motor outputs from their state,
- virtual motor nodes do not send outgoing graph edges to the network,
- structural evolution may add incoming chemical edges into virtual motor nodes,
- structural evolution may add outgoing chemical edges from virtual sensory nodes,
- gap-junction edits on virtual nodes are out of scope for v1 unless trivial to support.

This design makes the novel organ explicit while keeping the environment interface simple.

---

## 10. Controller design

## 10.1 CTRNN equation
Use a straightforward CTRNN implementation. The default form should be close to:

```text
x_dot = (-x + W_chem @ tanh(x) + gap_term(x) + b + u_clamp) / tau
```

Where:
- `W_chem` is the signed chemical matrix,
- `gap_term(x)` is symmetric diffusive coupling from the gap matrix,
- `b` is bias,
- `tau` is positive per-node time constant,
- `u_clamp` represents the virtual sensory node clamping or injection.

Use a simple explicit integrator for v1. Euler is acceptable.

## 10.2 Required controller methods
Implement at least:
- `reset(seed: int | None = None)`
- `step(observation: np.ndarray) -> np.ndarray`
- `serialize_state()`
- `load_state()`
- `apply_structural_genome()`
- `apply_hybrid_genome()`

## 10.3 State/output mapping
- Observations are normalized first.
- Virtual sensory node states are overwritten or clamped each control step.
- Wing commands come from the states of `VM_WING_L` and `VM_WING_R` after squashing with `tanh` or clipping.
- Action range in the environment is `[-1, 1]` per wing.

## 10.4 Baseline controller
Implement one simple fixed-connectome baseline:
- fixed connectome core,
- fixed sensor injection,
- trainable linear readout from designated motor nodes or virtual motor nodes.

This baseline exists only for comparison.

---

## 11. Environment design: `WingedWorm3DEnv`

## 11.1 Core properties
- MuJoCo-based 3D environment.
- Deterministic reset when given a seed.
- Separate physics step and controller step.

Recommended defaults:
- `physics_dt = 0.005`
- `control_dt = 0.02`
- `n_substeps = 4`
- episode timeout around `8.0` to `10.0` seconds.

## 11.2 Observation vector
The observation must be a flat NumPy array and documented in code.
Use this default ordering:
- body up-vector or tilt representation,
- roll and pitch,
- roll rate and pitch rate,
- linear velocity `(vx, vy, vz)`,
- height above ground,
- left wing hinge angle,
- right wing hinge angle,
- left wing hinge velocity,
- right wing hinge velocity.

Keep the default observation size modest and stable.

## 11.3 Action vector
Action shape: `(2,)`
- action[0] = left wing drive
- action[1] = right wing drive

Map actions to either hinge torque or target actuator command. Pick one and keep it consistent.
Torque control is acceptable and simple.

## 11.4 Reset distribution
Default reset should randomize within narrow ranges:
- torso position near a nominal start height,
- small roll/pitch perturbation,
- small angular velocity perturbation,
- small forward velocity perturbation.

Use a fixed default start height and forward speed in config.

## 11.5 Termination conditions
Implement at least:
- crash below minimum height,
- excessive tilt beyond threshold,
- NaN state,
- timeout.

## 11.6 Reward components to log
At every step or episode aggregate, log:
- alive reward,
- forward component,
- height component,
- tilt penalty,
- angular-velocity penalty,
- control-effort penalty,
- terminal penalty.

---

## 12. Structural evolution is the main method

## 12.1 Primary representation
The primary evolved object is **not** a dense weight vector.
The primary evolved object is a **bounded edit list** applied to the base connectome graph.

Each individual starts from `G0` and produces `G = apply_edits(G0, edits)`.

## 12.2 Allowed edit operators in v1
Required:
- `add_chem`
- `del_chem`
- `flip_sign`

Optional but desirable:
- `add_gap`
- `del_gap`
- `retarget_chem`

Recommended restrictions:
- no self-loops,
- no edits that break virtual node rules,
- no edits outside the configured masks,
- delete only edges that exist,
- add only edges that are absent.

## 12.3 Edit budget
Use a strict max edit budget.
Default for mock connectome:
- `max_edits = 8`

Suggested schedule for later experiments:
- 4, 8, 12 edit budgets.

Keep the first working experiment at `max_edits = 8`.

## 12.4 Weight/sign policy for structure-only stage
To keep structure-only meaningfully structural, use **coarse discrete edge values** rather than unconstrained continuous weights.

Recommended value set for newly added chemical edges:
- `{-1.0, -0.5, +0.5, +1.0}`

For sign flip:
- multiply current chemical edge weight by `-1`.

For gap edges, if implemented:
- use a small positive value set only.

This keeps the main search variable graph structure plus coarse edge type, not dense continuous fitting.

## 12.5 Evolution algorithm for structure-only stage
Implement a simple, reliable mutation-based evolutionary algorithm first.
Do **not** implement full NEAT unless it becomes necessary.

Recommended default:
- `(mu + lambda)` evolution strategy or simple elitist GA,
- mutation-only in v1,
- no crossover required,
- initial population created from random edit lists,
- parent selection by top-k or tournament,
- offspring created by mutate-copy.

Required mutation operations on genomes:
- append one random valid edit,
- delete one existing edit,
- replace one edit,
- retarget one edit,
- change the discrete value/sign of one edit.

Conflict resolution rules must be deterministic. Example:
- later edits override earlier conflicting ones,
- duplicate deletes collapse,
- duplicate adds keep the last version,
- invalid edits are dropped during normalization.

## 12.6 Fitness definition
Fitness is episode performance minus edit complexity cost.
Recommended form:

```text
fitness =
    mean_episode_reward
    - alpha * edit_count
    - beta * structural_complexity_penalty
```

Keep `alpha` small enough that useful structure can still emerge.
The edit penalty must be explicit and logged.

## 12.7 Evaluation during evolution
Use multiple rollout seeds per candidate to reduce lucky solutions.
Recommended defaults for the first real run:
- `train_eval_seeds = 4`
- `final_eval_seeds = 8`

Use fewer seeds for smoke tests.

---

## 13. Structure-first hybrid stage
This stage is secondary but important.
It must only be implemented **after structure-only works**.

## 13.1 Principle
First discover a structural scaffold, then locally refine continuous parameters.
Do not open unconstrained full-network parameter fitting.

## 13.2 Allowed continuous refinement targets
Only allow continuous deltas for:
- node biases of nodes touched by edits or adjacent to edited edges,
- `log(tau)` of nodes touched by edits or adjacent to edited edges,
- scale factors on newly added or edited chemical edges,
- optional scale factors on virtual motor incoming edges.

Do not allow unrestricted dense training of all weights.

## 13.3 Hybrid optimizer
A simple and acceptable default is:
- structure discovered by mutation-based evolution,
- local continuous refinement by **CMA-ES** on the restricted parameter subset.

The hybrid stage must preserve the structural edit list in the run artifacts.

---

## 14. Baselines and ablations
The experiment is not complete without comparison conditions.
Implement these in priority order.

### Required
1. **PD / heuristic smoke baseline**
   - used only to validate physics and reward.
2. **Fixed-connectome baseline with simple readout**
   - no graph edits,
   - minimal trainable output mapping.
3. **Structure-only evolution**
   - main experimental method.

### Strongly recommended
4. **Structure-first hybrid**
   - structure-only best candidate plus local continuous refinement.

### Optional if time allows
5. **Parameter-only baseline**
   - no graph edits,
   - continuous tuning of a parameter subset of similar size to the hybrid stage.
6. **Random-graph baseline**
   - graph with matched size statistics but no biological prior.
7. **Degree-preserving rewired baseline**
   - useful if real connectome mode is used later.

---

## 15. Run artifacts and outputs
Every train or eval command must write a self-contained run directory.

Required files inside each run directory:
- `config_resolved.yaml`
- `manifest.json`
- `metrics.csv`
- `generation_summary.csv`
- `best_genome.json`
- `best_graph_before_after.json`
- `best_graph_before.gml`
- `best_graph_after.gml`
- `best.ckpt`
- `last.ckpt`
- `eval_summary.json`
- `trajectory_best.npz`

Desirable optional artifacts:
- `best_episode.mp4`
- `edit_frequency.csv`
- `fig_learning_curve.png`
- `fig_best_vs_baseline.png`
- `fig_edit_histogram.png`
- `report.md`

All metrics files must be readable without the original Python process.

---

## 16. Analysis requirements
The repository is not done until it can summarize the experiment.

Implement analysis scripts that can generate at least these outputs:
- best fitness over generations,
- best reward over generations,
- baseline vs evolved comparison table,
- list of edits in the best genome,
- edit counts by type,
- graph summary before vs after,
- success rates under perturbations.

Minimal analysis functions:
- `summarize_run(run_dir)`
- `compare_runs(run_dirs)`
- `export_best_edit_table(run_dir)`

The best genome report must answer:
- how many edits were used,
- which nodes were most frequently edited,
- whether edits concentrated near sensory input, recurrent core, or motor output.

---

## 17. Config-driven behavior
Everything should be configurable from YAML.
Do not hardcode experiment choices deep in the source tree.

Required config sections:
- `experiment`
- `env`
- `controller`
- `evolution`
- `logging`
- `seeds`

### 17.1 Canonical mock experiment config
Implement a config equivalent to the following:

```yaml
experiment:
  name: exp001_mock
  mode: structure_only
  connectome_mode: mock

env:
  episode_seconds: 8.0
  physics_dt: 0.005
  control_dt: 0.02
  start_height: 1.5
  start_forward_speed: 1.0
  crash_height: 0.08
  max_tilt_deg: 75
  wind_gust_std: 0.0

controller:
  type: ctrnn
  tau_init: 1.0
  bias_init: 0.0
  use_virtual_nodes: true

evolution:
  population_size: 32
  elite_count: 4
  generations: 40
  max_edits: 8
  train_eval_seeds: [0, 1, 2, 3]
  mutation:
    p_append: 0.35
    p_delete: 0.15
    p_replace: 0.20
    p_retarget: 0.15
    p_rescale: 0.15
  new_edge_values: [-1.0, -0.5, 0.5, 1.0]
  edit_penalty: 0.05

logging:
  save_video: false
  save_trajectories: true
  save_graph_exports: true

seeds:
  master_seed: 123
```


Provide a second config for a longer run, for example `population_size = 96`, `generations = 150`.

---

## 18. Implementation order for Codex
Codex must implement in this order.
Do not jump ahead unless a small helper is necessary.

### Milestone 1 — Project scaffold and environment setup
Deliver:
- repository structure,
- requirements files,
- Dockerfile,
- docker-compose,
- README stub,
- `smoke_mujoco.py`.

Acceptance:
- `python scripts/smoke_mujoco.py` runs,
- MuJoCo can step a trivial model,
- tests can import package modules.

### Milestone 2 — Mock connectome loader and graph utilities
Deliver:
- bundled mock CSVs,
- `ConnectomeData` loader,
- graph mask generation,
- graph export helpers.

Acceptance:
- loader reads mock data,
- masks are valid,
- export/import round-trip works,
- tests verify matrix shapes and symmetry of gap weights.

### Milestone 3 — CTRNN core
Deliver:
- controller implementation,
- virtual node support,
- serialization,
- `apply_structural_genome()`.

Acceptance:
- controller steps deterministically,
- structural edits modify the compiled graph as intended,
- tests cover add/delete/sign-flip behavior.

### Milestone 4 — `WingedWorm3DEnv`
Deliver:
- MuJoCo XML generation or static XML asset,
- environment reset/step,
- aerodynamic force function,
- reward logging.

Acceptance:
- environment runs for at least 200 steps without controller-induced NaNs,
- random actions produce finite trajectories,
- PD smoke baseline can influence body behavior.

### Milestone 5 — Fixed-connectome baseline
Deliver:
- simple readout baseline,
- train/eval loop,
- run directory outputs.

Acceptance:
- baseline training/evaluation scripts complete,
- run artifacts are written,
- replay artifact is generated.

### Milestone 6 — Structure-only evolution
Deliver:
- structural genome,
- mutation operators,
- selection loop,
- fitness aggregation across seeds,
- checkpointing.

Acceptance:
- a short run completes from config,
- best genome contains explicit edits,
- graph before/after files are saved,
- metrics show generation progress.

### Milestone 7 — Analysis and report generation
Deliver:
- report script,
- comparison plots,
- edit summary export.

Acceptance:
- `make_report.py` produces a readable markdown summary and core figures for one run.

### Milestone 8 — Optional structure-first hybrid
Deliver:
- restricted continuous refinement stage,
- CMA-ES wrapper,
- comparison against structure-only.

Acceptance:
- hybrid stage can load the best structural candidate,
- local refinement runs and logs deltas,
- comparison summary is produced.

---

## 19. Validation commands
The repository should expose simple commands through `Makefile`.

Required targets:
- `make install`
- `make test`
- `make smoke`
- `make smoke-exp001`
- `make train-exp001`
- `make eval-latest`
- `make report-latest`

Expected command behavior:

```bash
make install          # install dev requirements
make test             # run pytest
make smoke            # smoke MuJoCo and mock connectome
make smoke-exp001     # very short 2–3 generation structure-only run
make train-exp001     # canonical structure-only run on mock connectome
make eval-latest      # evaluate latest/best checkpoint
make report-latest    # generate markdown report and figures
```

If Docker is implemented, also provide:
- `make docker-build`
- `make docker-smoke`
- `make docker-train-exp001`

---

## 20. Testing requirements
Tests should be lightweight and deterministic.

Minimum required tests:
- connectome mock loader test,
- matrix mask validity test,
- CTRNN deterministic step test,
- structural edit apply/remove test,
- environment smoke step test,
- reward component presence test,
- short evolution smoke test with tiny population and 2–3 generations.

Do not create flaky long-running tests.

---

## 21. Coding style and implementation rules
- Use type hints for public functions.
- Use dataclasses or Pydantic only if it helps clarity. Prefer dataclasses by default.
- Avoid premature abstraction.
- Keep most numerical code in NumPy.
- Separate pure graph-edit logic from MuJoCo environment code.
- Keep evolutionary code serializable and pickle-safe.
- Prefer simple JSON/CSV outputs over opaque binary formats, except for NumPy trajectory arrays and checkpoints.
- When a design is ambiguous, choose the option that makes structure-only evolution easier to analyze.

Do not:
- hide graph edits inside a dense tensor diff,
- train a generic neural controller and call it connectome-based,
- add a large dependency without necessity,
- skip analysis/export because “the training ran.”

---

## 22. Practical shortcuts allowed in v1
These shortcuts are explicitly allowed:
- using a small mock connectome instead of the full real worm graph,
- assuming positive base chemical weights if sign is missing,
- using simple quasi-steady lift/drag approximations,
- using mutation-only evolution instead of crossover-heavy NEAT,
- using graph-edit count as the structural distance proxy,
- evaluating robustness on a small perturbation set instead of a large benchmark.

These shortcuts are allowed because the main goal is to make **one structure-evolution experiment** runnable and analyzable.

---

## 23. What “done” means for this repository
The repository is done for v1 when a new user can:
1. clone the repo,
2. install dependencies or start Docker,
3. run a smoke test,
4. run `exp001_mock`,
5. inspect the best structural edits,
6. compare the evolved solution against the fixed-connectome baseline,
7. render or replay a best episode,
8. read a generated markdown report summarizing the result.

If these eight steps work, the repository is good enough to support one real research attempt.

---

## 24. Primary references for humans
These references are included so the project intent remains grounded. Codex does not need to fetch them during normal implementation.

1. Cook SJ et al. (2019). *Whole-animal connectomes of both C. elegans sexes*. Nature.
2. Witvliet D et al. (2021). *Connectomes across development reveal principles of brain maturation*. Nature.
3. Randi F et al. (2023). *Neural signal propagation atlas of Caenorhabditis elegans*. Nature.
4. Zhao M et al. (2024). *An integrative data-driven model simulating C. elegans brain-body-environment interactions*. Nature Computational Science.
5. Suárez LE et al. (2024). *Connectome-based reservoir computing with the conn2res toolbox*. Nature Communications.
6. Gleeson P et al. (2018). *c302: a multiscale framework for modelling the nervous system of C. elegans*. BMC Neuroscience.
7. Yim H et al. (2024). *Comparative connectomics of dauer reveals developmental plasticity*. Nature Communications.
8. Liao CP et al. (2024). *Experience-dependent, sexually dimorphic synaptic connectivity defined by sex-specific cadherin expression*. Science Advances.
9. Stanley KO, Miikkulainen R. (2002). *Evolving neural networks through augmenting topologies*. Evolutionary Computation.
