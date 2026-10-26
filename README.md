# pufferflow

- Built on pufferlib (documentation is at [puffer.ai](https://puffer.ai)).

## Commands

`checkpoints/town01.bin` is the Town01 demo policy (run 173, 512×3, score 0.792). `config/drive.ini` is 512 wide and 3 layers, matching this file, so neither command needs extra network flags. Both play Town01 (`data/mixed/binaries/map_001.bin`). Training also uses Town02 (`map_002.bin`) and the bottleneck (`map_000.bin`), 128 cars per copy. Those three binaries have to be present or the player exits before the window opens.

Open it on the local display:

```
./puffer eval checkpoints/town01.bin
./drive checkpoints/town01.bin
```

`./puffer eval` is the trainer and draws one map copy. `./drive` is the CPU player.

Score one headless episode:

```
./drive --headless --eval_episodes=1 checkpoints/town01.bin
```
