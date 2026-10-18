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

`checkpoints/bottleneck_2m.bin` is the 512×3 policy (trial 37, goal score 0.761):

```
./drive checkpoints/bottleneck_2m.bin --policy.hidden_size=512 --policy.num_layers=3
```

`checkpoints/bottleneck_1p2m.bin` is the 256×5 policy (run 89, goal score 0.789):

```
./drive checkpoints/bottleneck_1p2m.bin --policy.hidden_size=256 --policy.num_layers=5
```

`checkpoints/bottleneck_0p8m.bin` is the 256×3 policy (run 229, goal score 0.873):

```
./drive checkpoints/bottleneck_0p8m.bin --policy.hidden_size=256 --policy.num_layers=3
```
