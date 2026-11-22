# I-24 whole-scene macro evaluation

`velocity_mape.py` compares checkpoint rollouts with the full I-24-3D scene
clock, rather than the eight-second focal-vehicle window used by
`eval/dreal_highway`.

The bundled `rl_i24_*` checkpoints use a 128-unit, one-layer recurrent policy.
Those are the command defaults; the policy runner validates the checkpoint
weight count and fails if `--hidden` or `--layers` does not match.

- Scene 1 uses only the first 90 seconds of the source clock, matching the
  reliable spline interval used by `utils/i24.py`.
- Scenes 2 and 3 use their complete source clocks.
- Every eligible active vehicle is controlled by the checkpoint. There are no
  replayed or scripted background vehicles.
- Vehicles enter on their first source-valid frame and are removed immediately
  after their source-valid interval ends.
- Speed fields use 1 s × 15 m cells and are split by travel direction.
- Raw fields are per-cell sample averages. Smoothed fields convolve speed sums
  and sample counts separately with Gaussian sigma 1.5 s × 30 m, then divide.
- Empty cells are not zero speed. Velocity error uses common support; missing
  and extra support are reported separately.
- Conventional MAPE omits exactly-zero GT-speed cells and reports how many were
  omitted. A 1 m/s denominator-floor sensitivity metric, MAE, RMSE, and bias
  are included as well.

Build the 512-slot macOS/Linux evaluation runtime and run a small smoke test:

```bash
eval/macro_metrics/build_runtime.sh
python -m eval.macro_metrics.velocity_mape \
  --scenes 2 --rollouts 1 \
  --checkpoints checkpoints/rl_i24_coef0.bin \
  --output-dir analysis/macro_metrics/smoke
```

Run the agreed evaluation (two checkpoints, three scenes, 32 sampled rollouts):

```bash
python -m eval.macro_metrics.velocity_mape
```

Rollouts are independent and process-parallel. By default the evaluator uses
`min(rollouts, available CPUs)` workers. Set the concurrency explicitly when a
machine has shared or memory-constrained CPUs:

```bash
python -m eval.macro_metrics.velocity_mape \
  --scenes 3 --rollouts 32 --workers 32 \
  --checkpoints checkpoints/rl_i24_coef0p068.bin \
  --output-dir analysis/macro_metrics/scene3_coef0p068_32
```

Rollout `r` uses policy and simulator seed `--seed + r`. Every worker writes an
atomic trajectory archive before reporting success. Re-running the same command
resumes those archives and rebuilds fields, metrics, and figures without policy
inference. Use `--no-resume` only when intentionally replacing all requested
rollouts.

The default output directory is
`analysis/macro_metrics/i24_velocity_mape`. It contains:

- `summary.json`: protocol, scene eligibility, per-rollout metrics, and
  aggregate distributions;
- `rollout_metrics.csv`: pooled per-rollout metrics;
- `velocity_mape_cdfs.png`: raw and smoothed MAPE ECDFs;
- one compressed field archive and one best/median/worst figure for every
  scene/checkpoint pair;
- `rollouts/sceneN/CHECKPOINT/rNNN.npz`: reusable trajectories, headings,
  speeds, validity, collision state, and action bins for each rollout.

Field archives include both simulated field values and per-cell sample counts,
so later speed-, support-, occupancy-, and density-field metrics do not require
another rollout. The per-rollout archives support later detector counts,
headways, lane diagnostics, and other trajectory-derived metrics.

Best, median, and worst are ranked by pooled smoothed MAPE. The same rollout
indices are shown in raw and smoothed panels so smoothing does not silently
change which realization is being inspected.

## Diagnostic rollout video

`diagnostic_video.py` reproduces a selected stochastic rollout, saves its
vehicle trajectories, and renders a fixed bird's-eye comparison. The roadway
is split into three readable longitudinal strips. Simulated vehicles are solid
blue boxes and their same-frame ground-truth poses are green outlines.

For example, reproduce Scene 3 rollout 1 and render westbound at real-time 5
FPS (every other 10 Hz state):

```bash
python -m eval.macro_metrics.diagnostic_video \
  --scenario analysis/macro_metrics/scene3_wb_8/full_scene_data/json/i24_full_scene3.json \
  --data-dir analysis/macro_metrics/scene3_wb_8/full_scene_data \
  --checkpoint checkpoints/rl_i24_coef0.bin \
  --target-rollout 1 \
  --seed 101 \
  --trajectory-out analysis/macro_metrics/scene3_wb_8/scene3_rl_i24_coef0_r01_trajectory.npz \
  --video-out analysis/macro_metrics/scene3_wb_8/scene3_rl_i24_coef0_r01_wb_diagnostic.mp4 \
  --metric-label "Best smoothed WB MAPE: 291.8%"
```

Each rollout uses the deterministic seed `--seed + rollout index`, so rollout 1
can be reproduced directly without first running rollout 0. Once the trajectory
archive exists, presentation-only changes are fast: add `--render-only` to skip
simulation and re-encode directly from the saved trajectories.

The complete run is compute-heavy because each of the 192 rollouts advances
every active vehicle through the full scene using the native recurrent policy.
Use `--workers`, `--rollouts`, `--scenes`, and `--checkpoints` for incremental
runs.

The saved field arrays retain the physical longitudinal-position axis for the
next evaluation stage. A later shift-tolerant wave metric should remove or
optimize over *time* phase only (for example, localized temporal spectra at
each position); it should not use a global 2-D FFT magnitude that also erases
spatial location.
