# pufferflow

- Built on pufferlib (documentation is at [puffer.ai](https://puffer.ai)).

## Commands

Score `checkpoints/bottleneck_1m.bin` (256 wide, 4 layers) on `data/bottleneck/binaries/map_000.bin`:

```
./drive --headless --eval_episodes=1 checkpoints/bottleneck_1m.bin
```

Open it in the GUI window:

```
./drive checkpoints/bottleneck_1m.bin
```

## I24-3D to spiced self-play prototype

`ocean/drive/i24_3d_to_spiced.py` converts the 718 hand-annotated I24-3D
splines without requiring any extra Python packages. It writes both a compact,
full-length 10 Hz intermediate and 91-state JSON scenarios for
`spiced_self_play`. Each exported scenario also gets a self-contained animated
HTML preview under `previews/`; open it directly in a browser.

From this repository, generate one scene-2 proof of concept:

```bash
python ocean/drive/i24_3d_to_spiced.py \
  --i24-root ../i24_3d_dataset/I24-3D \
  --output-dir /tmp/i24_spiced_prototype \
  --scene 2
```

Use `--num-windows 0` to export every non-overlapping 9-second window for the
selected SDC, or `--sdc-id ID` to select a particular source trajectory.

Generate the complete three-scene batch for spiced self-play with:

```bash
python ocean/drive/i24_3d_to_spiced.py \
  --i24-root ../i24_3d_dataset/I24-3D \
  --output-dir data/i24_spiced_batch \
  --all-scenes \
  --all-sdcs \
  --num-windows 0 \
  --skip-preview \
  --skip-full
```

This produces 1,683 non-overlapping scenarios from the 639 trajectories that
contain at least one complete 91-state window. Add `--exclude-major-flags` to
omit SDCs with extreme speed or calibrated-range warnings, producing 1,571
scenarios from 586 trajectories. Background vehicles with warnings are kept as
context in either mode. The aggregate `manifest.json` records every output and
every exclusion.

Heading stabilization is optional and disabled by default. To produce a
separate clean batch with confidence-gated headings while preserving the
original batch, use a sibling output directory:

```bash
python ocean/drive/i24_3d_to_spiced.py \
  --i24-root ../i24_3d_dataset/I24-3D \
  --output-dir data/i24_spiced_batch_clean_stabilized \
  --all-scenes \
  --all-sdcs \
  --num-windows 0 \
  --exclude-major-flags \
  --stabilize-headings \
  --skip-preview \
  --skip-full
```

With `--stabilize-headings`, orientation is estimated by linear regression over
a centered 1.0-second position window. Below 0.5 m/s, the exporter holds the
last reliable orientation (or backfills the first reliable one at track start).
Positions and velocities are unchanged. The scenario metadata and aggregate
manifest record whether stabilization was enabled and its exact parameters.

Use a fresh output directory when changing export options; the prototype
overwrites matching files but does not delete older scenario files. The 79
source trajectories shorter than 9.0 seconds remain available in the optional
full-length intermediates but cannot be complete SDC windows.

The prototype's road features are inferred from I24-3D's 12-foot homography
grid. They are sufficient to exercise the conversion path, but they are not a
surveyed I-24 map. The spiced simulator still requires its normal dependency
installation before its `save_map_binary` step can turn the scenario JSON into
`.bin` files.

### I24-3D speed fields

Generate longitudinal speed fields from the ground-truth trajectories with:

```bash
python ocean/drive/plot_i24_speed_field.py
```

The script bins the 10 Hz samples on a regular time-position grid, computes a
sample-weighted mean speed in each cell, and optionally smooths the speed sums
and sample counts before dividing them. By default it writes direction and
Scene 3 lane views under `figures/i24_speed_field/`. Use
`--time-bin-seconds`, `--space-bin-meters`, `--smooth-time-seconds`, and
`--smooth-space-meters` to control the field resolution and smoothing; set both
smoothing scales to zero to see only directly observed cells.

Generate net bumper-to-bumper distance-headway distributions and lane-assignment
diagnostics with:

```bash
python ocean/drive/plot_i24_headway_distributions.py
```

Scenes 1 and 3 use the calibrated lane centers and flag vehicle footprints that
overlap an internal boundary. Because snowy Scene 2 traffic follows visible
ruts rather than the calibration grid, that scene instead infers curved
effective-stream centerlines from the smoothed trajectories. Its proposed
leader must also overlap the follower laterally after both offsets are measured
relative to the shared centerline. The script writes the six headway
distributions as both histograms and empirical CDFs, a Scene 2 centerline audit,
a calibrated-versus-effective Scene 2 comparison, and a JSON analysis under
`figures/i24_headways/`. The effective-stream smoothing, density-peak,
confidence, and overlap thresholds are exposed as `--effective-*` options;
`--cdf-max-gap-meters` controls the CDF display range.

Generate total-roadway throughput visualizations from exact trajectory
crossings with:

```bash
python ocean/drive/plot_i24_throughput.py
```

The script does not assign lanes. Its high-fidelity view interpolates every
vehicle crossing at virtual cross-sections spaced 25 m apart and reports flow
in complete 10-second intervals. Its ideal-loop view places perfect detector
stations at 25%, 50%, and 75% of the calibrated longitudinal range and reports
complete 30-second interval counts. Outputs under `figures/i24_throughput/`
include the dense position-time flow field, cumulative station counts,
ideal-loop interval counts, fine-versus-coarse temporal aggregation, observed
segment accumulation, and a JSON summary. Gate spacing and both aggregation
intervals are configurable from the command line.

To replay the recorded human ego and surrounding traffic exactly in the
PufferDrive window, use logged initialization plus `replay_logged_agents`:

```bash
./drive checkpoints/bottleneck_1m.bin \
  --env.data_dir=data/i24_congested_v179w1 \
  --env.map_id=0 \
  --env.agent_initialization=0 \
  --env.replay_logged_agents=1
```

The checkpoint is still loaded by the viewer, but its actions are ignored in
replay mode. Replay uses the ego trajectory's stored length, loops at its end,
and returns to the first logged frame when Reset is pressed.

Export and compile one complete source trajectory for PufferDrive replay with:

```bash
python ocean/drive/i24_3d_to_spiced.py \
  --scene 3 \
  --sdc-id 179 \
  --full-length \
  --output-dir data/i24_congested_v179_full \
  --skip-full \
  --skip-preview

python ocean/drive/dataset.py \
  --data_folder data/i24_congested_v179_full/scenarios \
  --output_dir data/i24_congested_v179_full/binaries \
  --num_workers 1

./drive checkpoints/bottleneck_1m.bin \
  --env.data_dir=data/i24_congested_v179_full \
  --env.map_id=0 \
  --env.agent_initialization=0 \
  --env.replay_logged_agents=1
```

`--full-length` is intended for visualization and exact PufferDrive replay;
the current spiced self-play training path still consumes 91-state scenarios.
Its `--max-objects` setting (32 by default) selects context vehicles based on
their closest approach anywhere along the full ego trajectory. Add
`--stabilize-headings` to a full-length export when a separate stabilized
visualization dataset is desired; use a new output directory to retain both.

`checkpoints/bottleneck.bin` is the 512×5 policy:

```
./drive checkpoints/bottleneck.bin --policy.hidden_size=512 --policy.num_layers=5
```

`checkpoints/bottleneck_2m.bin` is the 512×3 policy (trial 37, goal score 0.761):

```
./drive checkpoints/bottleneck_2m.bin --policy.hidden_size=512 --policy.num_layers=3
```
