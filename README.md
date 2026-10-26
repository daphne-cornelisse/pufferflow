# pufferflow

- Built on pufferlib (documentation is at [puffer.ai](https://puffer.ai)).

## Commands

`checkpoints/town01.bin` is the Town01 demo policy (run 173, 512×3, score 0.792). `config/drive.ini` is 512 wide and 3 layers, matching this file. `./drive` plays Town01 (`data/mixed/binaries/map_001.bin`). Training also uses Town02 (`map_002.bin`) and the bottleneck (`map_000.bin`), 128 cars per copy.

Open it in the GUI window:

```
./drive checkpoints/town01.bin
```

Score one headless episode:

```
./drive --headless --eval_episodes=1 checkpoints/town01.bin
```
