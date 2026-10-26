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

Drive the Waymo self-driving car against a frozen behavioral-cloning anchor. WASD or the arrow keys hold an acceleration bin and a steering bin. The other cars replay the log, and the scene returns to the first frame every 90 steps. The panel shows the anchor distribution for the SDC observation, then KL(anchor || act), KL(act || anchor), and the 1-Wasserstein distance. "Let anchor take control" rolls out the checkpoint. "New scene" at the top right loads another intersection.

```
./drive --anchor checkpoints/bc_womd_100.npz
```

That loads a dense intersection from `data/womd`. `--env.map_id` picks one of the ten curated maps when the id is one of them.
