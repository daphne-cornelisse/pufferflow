# pufferflow

- Built on pufferlib (documentation is at [puffer.ai](https://puffer.ai)).

## Commands

Score `checkpoints/bottleneck.bin` on `data/bottleneck/binaries/map_000.bin`:

```
./drive --headless --eval_episodes=1 checkpoints/bottleneck.bin
```

Open the same checkpoint in the GUI window:

```
./drive checkpoints/bottleneck.bin
```

`config/drive.ini` `control_mode = precal_idm` drives every vehicle with SUMO's calibrated IDM instead of the learned policy. `policy` is the bicycle model.
