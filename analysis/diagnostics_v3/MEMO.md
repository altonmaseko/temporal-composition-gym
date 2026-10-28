# Diagnostics after protocol v2: decision memo (Prompt 2)

Probes: [run_probes.py](run_probes.py), outputs in [out/](out/). Single diagnostic seed (training seed
401, train-env seed 6,100,401, 50 greedy evaluation episodes on seeds 6,000,000–6,000,049), CPU,
baseline hyperparameters, no baseline code, results or plan files touched. One seed per arm, so
read these as directional, not as estimates.

## 1. What the probes showed

**Delivery, DQN-TS, 300k steps** (baseline DQN-TS, first 300k steps, 5 seeds: 8.4 training successes per run, 1.6% success in the last 50k, greedy eval 0%)

| Arm | Training successes (rate in first / last 50k) | Deliveries per training ep (last 50k) | Training causes (last 50k) | Greedy eval, 50 ep |
|---|---|---|---|---|
| `num_packages=1` | 8 (0.8% → **0%**) | 0.05 | lava 43%, timeout 33%, premature goal 24% | 0% success, 0 deliveries, 50/50 timeout |
| `percent_death_squares=0` | 16 (1.6% → 0.8%) | 1.26 | timeout 60%, premature goal 38% | 0% success, 0.08 deliveries, 50/50 timeout |

**Neither shortening the chain nor removing lava is enough.** With one package the learner loses
even the success rate of random play. Without lava, death simply shifts to the premature-goal trap,
and the greedy policy still loops (timeouts). In 300k steps the barrier is the learner and the
evaluation rule (2-cycles and policy churn, baseline REPORT §4.1), not the task's length or hazards.
This does not justify any delivery task change.

**Maintenance, PPO-RM, γ = 0.999, 500k steps** (completed: 499,712 steps in 12.4 min, not stopped early)

| | γ = 0.999 probe (1 seed) | Baseline PPO-RM γ = 0.99, same steps (5 seeds) |
|---|---|---|
| Services (successful) per 100k training steps | **0** in every window | 0.2–1.4 (mean over seeds) |
| Successes during training | 0 / 140 episodes | 0 / 689 |
| Training return per episode, 100k–500k | −255 to −458 | −1,123 to −1,754 |
| Stage exits per 100k (after the first 100k) | 1–5 (of which 0–4 with an uncollected code, −1,000) | uncollected-code exits 1.6–3.4 (clean exits not logged) |
| Greedy eval (γ probe at 500k vs baseline snapshot at 409.6k, same 50 seeds) | stage exits **0.52**/ep, codes 0.74/ep, services 0, success 0, return −2,463 (14/50 exit with an uncollected code) | stage exits 0.12/ep, codes 0.46/ep, services 0, success 0, return −1,702 |

At a quarter of the budget, γ = 0.999 changes the behaviour as expected. The agent avoids the
boundary (training return 4× better) and its greedy policy moves through stages 4× more often.
It does **not** yet produce a single service or success, and it exits stages before collecting the
code (−1,000). γ alone is necessary under the objective argument but not shown to be sufficient
at 500k steps.

**Colour, DQN-TS, `reward_density="dense"`, 500k steps: GPU only, not run.** On a Vast.ai machine
set up as in PRODUCTION_LAUNCH_v2 §4 (steps A, venv `~/tcg`), copy the probe script (it is not
committed), then run it in tmux:

```powershell
scp -P <PORT> "C:\Users\alton\Desktop\School Work\Research Project Development\NEW-temporal-composition-gym\analysis\diagnostics_v3\run_probes.py" root@<HOST>:~/run_probes.py
```
```bash
mkdir -p ~/temporal-composition-gym/analysis/diagnostics_v3 && mv ~/run_probes.py ~/temporal-composition-gym/analysis/diagnostics_v3/ && cd ~/temporal-composition-gym && WANDB_MODE=disabled ~/tcg/bin/python -B analysis/diagnostics_v3/run_probes.py --probe colour_dense --steps 500000 --device cuda --max-minutes 40
```
Expected time on an RTX 3090: about **11–13 min** of training (baseline DQN-TS colour ran at 780
steps/s) plus about 1 min of evaluation. RAM about 6 GB (SB3 image replay). The result is
`analysis/diagnostics_v3/out/colour_dense.json`; bring it back with `scp`. Read it against the
baseline DQN-TS at 500k: 0.6% training success and greedy 0%.

## 2. Proposed protocol v3 (minimal; each change applies to every algorithm in its environment)

| # | Change | Applies to | Evidence | Status |
|---|---|---|---|---|
| 1 | **Maintenance γ = 0.999** (effective horizon 1,000 steps ≈ 17 s, matched to the 60 Hz control rate and the oracle's ~1,400-step solution) | PPO, Recurrent PPO, PPO-RM | Under γ = 0.99 the oracle's discounted return is −4.9, below hovering (−2 to −10), so the objective does not prefer solving the task. At γ = 0.999 the oracle scores +287. The probe confirms the behaviour shift (boundary avoided, 4× more stage exits), with no successes yet at 500k. | Recommended. This fixes the objective, not an algorithm. |
| 2 | **Secondary evaluation with the training-time action noise** (ε = 0.05 for DQN-family, `deterministic=False` for PPO-family), reported next to the greedy headline on the same seeds | all algorithms, all envs | Frozen greedy delivery and colour policies fall into 2-cycles in 100% of rollouts. ε = 0.05 recovers 0.3–0.7 deliveries per episode against 0.04–0.3 greedy. The headline rule stays greedy unless you decide otherwise (§3). | Recommended (reporting only; no training change) |
| 3 | **Logging**: SB3 DQN mean Q and gradient norm per interval; per-episode counts of stage exits, codes and service attempts (maintenance) and wall touches (colour) in `train_episodes.csv` | all | Q inflation (delivery 12–28 against a realised 0; QRM-Cue nav heads up to 868) was visible only through snapshots; maintenance stage exits were not logged | Recommended (no semantic change) |

Not proposed, and why:
- **Delivery task changes** (fewer packages, no lava): the probes show neither helps.
- **Colour dense reward / shorter sequences**: not justified until the GPU probe shows dense reward makes the task learnable. Even then it is a research-scope choice (below).
- **PPO entropy bonus, reward normalisation, Double-DQN, QRM-Cue stabilisation**: these are learner settings, so they belong in tuning, applied per algorithm family within the tuning budget, not in the protocol.

## 3. Decisions that are yours (research scope)

1. **Headline evaluation rule:** keep greedy (strict, comparable to the literature, but penalises
   memoryless value-based agents through loops), or make the noisy rule the headline.
2. **Colour reward density:** dense (+1 per correct wall) changes the credit-assignment problem
   that RQ2 studies (difficulty vs delay). Decide after the GPU probe whether sparse colour stays
   the main condition, with dense as an ablation, or the reverse.
3. **Maintenance beyond γ:** if γ = 0.999 still yields no services at the full 2.048M budget, the
   remaining options (reward shaping toward codes/stations, a longer budget, action repeat) all
   change the task's difficulty and are your call.
4. **Budgets:** with every cell near 0%, whether 1M / 3M / 2.048M is enough is open. A longer
   budget is a resource decision.

## 4. Should tuning start from v3?

**Yes, but only after decisions 1–2 and with v3 change 1 in place.** Tuning maintenance on v2
would optimise an objective under which the solution is not optimal. Delivery and colour tasks
are unchanged in v3, so their tuning can start as soon as the evaluation rule (decision 1) is
fixed, because the tuning metric depends on it. Colour tuning should wait for the dense-reward
GPU probe (≈ 15 min). Version the change log as v3 before any tuning seed (101–199) is used.
