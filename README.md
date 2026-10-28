# pufferflow

- Built on pufferlib (documentation is at [puffer.ai](https://puffer.ai)).

## Commands

`checkpoints/drive/town01_2b/0000002000158720.bin` is the Town01 demo policy (2 billion steps, 512×3, score 0.908). `config/drive.ini` is 512 wide and 3 layers, matching this file, so neither command needs extra network flags. Both play Town01 (`data/mixed/binaries/map_001.bin`). Training also uses Town02 (`map_002.bin`) and the bottleneck (`map_000.bin`), 128 cars per copy. Those three binaries have to be present or the player exits before the window opens.

Open it on the local display:

```
./puffer eval checkpoints/drive/town01_2b/0000002000158720.bin
./drive checkpoints/drive/town01_2b/0000002000158720.bin
```

`./puffer eval` is the trainer and draws one map copy. `./drive` is the CPU player.

Score one headless episode:

```
./drive --headless --eval_episodes=1 checkpoints/drive/town01_2b/0000002000158720.bin
```

Drive the Waymo self-driving car against a frozen behavioral-cloning anchor. WASD or the arrow keys hold an acceleration bin and a steering bin. On a Waymo scene the other cars replay the log, and the scene returns to the first frame every 90 steps. The panel shows the anchor distribution at the SDC observation. While you or the anchor drive, it reports the entropy of that distribution. A policy checkpoint links an RL policy; while that policy drives, the panel reports KL(anchor || policy), KL(policy || anchor), and the 1-Wasserstein distance. "New scene" at the top right loads another intersection, or CARLA Town01.

```
./drive --anchor checkpoints/bc_womd_1000.npz
```

That loads both `checkpoints/bc_womd_1000.npz` and the regularized RL policy `checkpoints/drive/town01_bc1000/0000000578813952.bin` (512×3, KL to the BC anchor, coefficient 0.075). Score 0.681, collision rate 0.906, completion rate 0.707, off-road rate 0.398, anchor loss 5.931. A following `.bin` replaces the policy. The scene is a dense intersection from `data/womd`. `--env.map_id` picks one of the ten curated maps when the id is one of them. `--env.map_id=1` loads CARLA Town01 (`data/mixed/binaries/map_001.bin`). That map has no trajectory, so each car's goal is sampled on the lane ahead of it. The policy uses `policy.hidden_size` and `policy.num_layers` from `config/drive.ini`.
