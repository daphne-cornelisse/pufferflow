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
