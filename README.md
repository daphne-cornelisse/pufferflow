# pufferflow

- Built on pufferlib (documentation is at [puffer.ai](https://puffer.ai)).

## Commands

`checkpoints/town01.bin` is the Town01 demo policy (run 173, 512×3, score 0.792). `config/drive.ini` is 512 wide and 3 layers, matching this file. The default map is Town01 (`data/carla`, map 0).

Open it in the GUI window:

```
./drive checkpoints/town01.bin
```

Score one headless episode:

```
./drive --headless --eval_episodes=1 checkpoints/town01.bin
```
