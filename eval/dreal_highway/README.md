# DREAL highway

I24 realism scores. The histograms are the WOSAC 2024 bins in `dreal.ini`.

```
python eval/dreal_highway/score.py
```

The script reads the three scenes in `data/i24` and the greedy anchor `analysis/offline/i24/bc_3.npz`. It writes `analysis/offline/i24/dreal_highway.json`.

A drive checkpoint is an extra controller on that same report. It does not replace the three, and it does not write `dreal_highway.json`.

```
python eval/dreal_highway/score.py \
  --policy coef0=checkpoints/i24_anchor/drive/i24_coef0/0000000999817216.bin \
  --policy coef0p1=checkpoints/i24_anchor/drive/i24_coef0p1/0000000999817216.bin \
  --out analysis/offline/i24/dreal_rl.json
```

Each checkpoint is 32 sampled closed loops. The action server is `/tmp/i24_policy_act`, the same PufferNet the player runs, with the recurrent state cleared on the first step of each loop.

## Required comparisons

Every run scores these three controllers on every scene, in this order. The report is written only after all three have a finite likelihood on the same scenes.

- **Ground truth.** The logged window, copied across 32 rollouts. This is the ceiling.
- **Discretized expert.** Closed-loop rollout of the stored action bins. A missing bin coasts.
- **Anchor.** Greedy closed-loop rollout of `bc_3.npz`.

## Window

Logged frame 10, then 80 more states (81 total). Agents are vehicles and trucks that are fit and valid at frame 10. Unfit agents stay parked. The integrator does not freeze a car that reaches its goal.

Features are computed on that one trajectory and repeated along the rollout axis, so each histogram still sees 32 × 81 samples.

## Score

The seven likelihoods are linear speed, linear acceleration, angular speed, angular acceleration, distance to the nearest object, time to collision, and signed distance to the road edge.

Each agent's likelihood is the exp of its mean log probability. The ground-truth ceiling for a metric is the mean of those agent likelihoods, and the reported value divides by that ceiling. Every score is the mean ± the sample standard deviation across agents, and the report gives the total number of agents. An agent is included when the ground-truth likelihood is finite on all seven metrics. Meta is the unweighted mean of the seven normalized likelihoods for each agent. ADE is the displacement from the log over the same window.

## Controllability

Reported beside the likelihoods, and not inside meta. The denominator is the scored agents. A sampled policy uses agents × rollouts, one episode per rollout.

A collision is the simulator's vehicle-box overlap. An off-road collision is the simulator's road-edge intersection. A run of overlapping steps counts as one event. The two rates are agents with at least one event in the episode, and total events divided by those agents. Ground truth ignores timesteps the log marks invalid.
