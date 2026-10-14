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

`checkpoints/bottleneck.bin` is the 512×5 policy:

```
./drive checkpoints/bottleneck.bin --policy.hidden_size=512 --policy.num_layers=5
```

`config/drive.ini` `control_mode = precal_idm` drives every vehicle with SUMO's calibrated IDM instead of the learned policy. `policy` is the bicycle model.
