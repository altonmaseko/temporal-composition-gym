# Protocol v2 baseline: analysis and validation (Prompt 1)

Scope: the 55 runs in `run_scripts/baseline/results/v2/full/` (commit `10334db`), read-only.
Everything here was produced by the scripts in this folder (see [Reproduce](#9-reproduce)).
Machine-readable summary: [summary.json](summary.json). References are the oracle / uniform-random
numbers in `run_scripts/baseline/validation/task_validation_v2.json`.

## 0. Bottom line

1. **The record is clean.** All 55 planned runs completed and verified on their first attempt.
   There were no resumes or failures. Each run's config hash equals the plan's, the budgets are
   exact, and the gradient-update counts match the analytic expectation. One code revision
   (`f772875`, code-identical to validated `7d94a93`) produced every run, on the planned devices.
   **No pipeline defect was found**: rewards, `terminated`/`truncated` flags, replay sequences
   and QRM counterfactual heads all reach the learners correctly (§5).
2. **Final success is ~0% in all 11 cells.** The only non-zero values are PPO maintenance
   (1 success in 500 final episodes) and isolated periodic points (PPO-RM 5%, QRM-Cue 2%, one
   seed each). Final returns sit at the level of "do nothing and avoid penalties": delivery
   0.4–1.4, colour −0.2–0, maintenance −1,650 to −2,920. The normalised return
   (0.07–0.63) measures **penalty avoidance, not task progress** (§2).
3. **The task-state controls fail too, so missing memory is not the binding constraint under
   these defaults.** The evidence points to three non-memory causes, each confirmed by a
   measurement (§4, §5):
   - **Maintenance: γ = 0.99 is too short for a 60 Hz task.** Under the training objective
     (γ = 0.99) the oracle's discounted return is **−4.9**, below that of hovering in place
     (−2 to −10). The first task reward arrives after about 670 steps and the +1,000 completion
     has weight 8×10⁻⁶. The objective does not favour solving the task. The policies collapse to
     a saturated action (no thrust, full torque; std 0.05–0.1) and rest on the boundary, or hover.
   - **Colour: a sparse +10 against −1 per wrong wall teaches "never touch a wall".** Greedy
     policies stop about one block short of a wall and oscillate there. The replay holds 0.3–4
     completion transitions per 100k, so at most 0.13% of minibatches see a success.
   - **Delivery: frozen greedy policies fall into 2-step loops** (100% of rollouts, cycle
     length 1–2, first repeat by step 3–5). The training episodes looked better (85–626
     successes per cell, about 1 delivery per episode) only because the weights change every
     4 steps. A frozen snapshot at the same ε makes 3–4× fewer deliveries, and switching between
     snapshots reproduces the training numbers. Even then ~45% of episodes end in lava.
4. **Value estimates are inflated everywhere and diverge in QRM-Cue.** Delivery Q(s₀) is 12–28
   while the realised greedy return is ~0. In colour, QRM-Cue's navigation heads exceed the
   largest possible return (+10) in 4/5 seeds, reaching 868 in seed 1. This is a confirmed fact;
   it is a learning-stability problem, not a coding error (§5.7).
5. **Nothing found invalidates the baseline as a record of protocol v2.** It cannot support an
   RQ1 ranking: every cell sits at the floor, and in maintenance the objective itself does not
   reward the task (§7).

## 1. Reconciliation

Script [01_reconcile.py](01_reconcile.py) → [out/reconcile.json](out/reconcile.json), [out/reconcile_runs.csv](out/reconcile_runs.csv).

| Check | Result |
|---|---|
| `check_manifest.py` re-run (output to `out/manifest_recheck.json`) | exit 0, **55/55 complete**, 1 attempt per planned run, 0 unexpected; states identical to committed `manifest.json` |
| `baseline_plan.json` vs plan code (`write --check`) | up to date |
| Config hash: recomputed from `config.resolved.json` = `identity.json` = plan = `final_summary.json`; full resolved config equals the plan's | 55/55 |
| Budget: `env_steps_done` = summary step = last eval step = budget (1,000,000 / 3,000,000 / 2,048,000) | 55/55 |
| Step accounting: Σ training-episode lengths = last episode's `global_step`; unfinished tail < episode limit; eval at 0, 5%, …, 100% (21 points); 200/100 final episodes; 50/20 periodic episodes per point | 55/55 |
| Gradient updates vs analytic expectation (SB3 DQN 247,500 / 745,000; custom 247,501 / 745,001; PPO family 10,000 epoch-passes) | 55/55 |
| Seeds: train seed ∈ {1..5}; env first-reset seed 10,000,000 + 1,000·seed; periodic / final episode seeds 1,000,000+i / 2,000,000+i, identical across all runs of an env | 55/55 |
| Seed independence: no two seeds of a cell share their first 50 training episodes; step-0 evaluations differ except in colour, where several untrained policies push one direction and produce identical (−20.5, 300) episodes | ok |
| Paired design: same seed → same first training layout across algorithms; DQN and DQN-TS share the action stream (identical first episode) | as designed |
| Code revision | all 55 runs `f772875`, `dirty: false`; `git diff 7d94a93 f772875` on code paths is empty (only docs, validation JSON and `play_human/`); no code change from `f772875` to HEAD |
| Software | Python 3.12.15 on all machines, one package set (torch 2.14.0, SB3 2.9.0, sb3-contrib 2.9.0, gymnasium 1.3.0) |
| Devices | colour on NVIDIA RTX 3090 (CUDA) in all 20 runs; delivery and maintenance on CPU (4 torch threads); three hosts, one batch each |
| Resumes / failed attempts | none: `resumes: []`, `restored_steps: 0`, no `metadata_resume_*` or `*.pre-resume-*` files; every `batch_status.tsv` exit code 0; machine logs end "All N runs exited 0" |

Throughput and wall-clock per run (mean of 5): delivery DQN 1,780 steps/s (0.16 h), DRQN 545 (0.51 h),
QRM 2,550 (0.11 h), DQN-TS 1,710 (0.16 h); colour DQN 860 (0.98 h), DRQN 228 (3.66 h),
QRM-Cue 1,180 (0.72 h), DQN-TS 780 (1.08 h); maintenance PPO 1,370 (0.42 h), RecurrentPPO 573
(1.00 h), PPO-RM 1,790 (0.32 h). Peak RSS: 6.0 GB (SB3 colour, replay stores obs + next obs),
4.1 GB DRQN colour, 3.8 GB QRM-Cue, < 1.1 GB otherwise.

## 2. Results against the references

Script [02_learning.py](02_learning.py) → [out/learning.json](out/learning.json), [out/final_by_run.csv](out/final_by_run.csv).
Final evaluation: 200 episodes per run (maintenance 100), 5 seeds. Return CI = 95% bootstrap over seeds.

| Env | Algorithm | Final success | Final return (95% CI) | Norm. return IQM | Progress / ep | Best periodic success (max seed) | Training-episode successes |
|---|---|---|---|---|---|---|---|
| delivery | DQN | 0.000 | 0.6 (0.3, 0.8) | 0.388 | 0.06 | 0.00 | 212 / 32,215 |
| delivery | DRQN | 0.000 | 1.4 (0.8, 1.8) | 0.394 | 0.31 | 0.00 | 172 / 34,840 |
| delivery | QRM | 0.000 | 0.6 (0.4, 0.8) | 0.388 | 0.07 | 0.00 | 85 / 33,688 |
| delivery | DQN-TS (control) | 0.000 | 0.4 (0.2, 0.5) | 0.385 | 0.04 | 0.00 | 626 / 33,387 |
| colour | DQN | 0.000 | −0.03 (−0.1, 0.0) | 0.075 | 0.00 | 0.00 | 219 / 50,052 |
| colour | DRQN | 0.000 | −0.15 (−0.3, 0.0) | 0.068 | 0.00 | 0.00 | 137 / 50,030 |
| colour | QRM-Cue | 0.000 | −0.11 (−0.3, 0.0) | 0.077 | 0.05 | 0.02 | 470 / 50,114 |
| colour | DQN-TS (control) | 0.000 | −0.02 (−0.1, 0.0) | 0.077 | 0.07 | 0.00 | 394 / 50,101 |
| maintenance | PPO | 0.002 | −2,285 (−3,294, −1,223) | 0.497 | 0.01 | 0.00 | 0 / 2,701 |
| maintenance | Recurrent PPO | 0.000 | −2,920 (−3,338, −2,154) | 0.388 | 0.00 | 0.00 | 0 / 2,641 |
| maintenance | PPO-RM | 0.000 | −1,648 (−2,744, −551) | 0.631 | 0.03 | 0.05 | 0 / 2,736 |
| reference | oracle (d / c / m) | 1.00 / 1.00 / 1.00 | 80.0 / 10.0 / 1,076 | 1 | 3 / 3–5 / 1–2 | | |
| reference | random (d / c / m) | 0 / 0 / 0 | −49.5 / −0.8 / −6,037 | 0 | | | |

Success AUC is ≤ 0.0005 in every cell. Final causes: delivery and colour are > 99% `timeout`;
maintenance is 87–95% `timeout`, and the rest are mostly `uncollected_codes_at_exit` (−1,000).

**The normalised return reads as penalty avoidance.** A policy that never moves scores 0.39
in delivery (return 0) and 0.08 in colour. In maintenance, hovering off-path scores about 0.8 and
resting on the boundary about 0.37. The spread in maintenance (seed dots 0.37–0.80 in
[figures/final_normalised_return.png](figures/final_normalised_return.png)) is exactly hover vs
rest (§4.3), not task progress. Do not rank cells on this metric.

Figures:
[eval curves delivery](figures/eval_curves_delivery.png) ·
[colour](figures/eval_curves_colour.png) ·
[maintenance](figures/eval_curves_maintenance.png) ·
eval causes [d](figures/causes_eval_delivery.png) [c](figures/causes_eval_colour.png) [m](figures/causes_eval_maintenance.png) ·
training causes [d](figures/causes_train_delivery.png) [c](figures/causes_train_colour.png) [m](figures/causes_train_maintenance.png) ·
training episodes [d](figures/train_episodes_delivery.png) [c](figures/train_episodes_colour.png) [m](figures/train_episodes_maintenance.png) ·
optimisation [d](figures/updates_delivery.png) [c](figures/updates_colour.png) [m](figures/updates_maintenance.png) ·
[greedy vs exploratory](figures/behaviour_greedy_vs_explore.png).

## 3. Learning diagnostics

### 3.1 Delivery (1M steps, ε 1 → 0.05 over 200k)
- **Random warm-up (first 10k steps, ε = 1):** 1,419–1,436 episodes per cell. About 19% contain
  at least one +10 delivery (the episode still ends at −40 to −50), and none succeeds. Causes:
  lava 69–76%, premature goal 22–30%.
- **Learning happens, but toward safety.** From about 200k steps, training return rises from
  −49 to about −22 and progress to 0.8–1.1 deliveries per episode, with 55–70% of episodes
  delivering at least once. The first training successes come at 12k–178k steps. Late training
  success rate: DQN-TS 5.3%, DQN 1.5%, DRQN 0.9%, QRM 0.7%. Even at ε = 0.05, about 45% of
  training episodes still end in lava and 25–31% time out.
- **Greedy evaluation diverges from training at once.** From the 50k point onward, evaluation is
  100% `timeout`, return ~0 and progress 0.04–0.3. Late training (ε = 0.05) averages return −22
  to −33 with 0.8–1.1 deliveries per episode.
- Optimisation is stable. DQN TD loss (last step) 0.03–0.4; DRQN/QRM loss 0.17–0.5 with gradient
  norms 1–6 (clip 10); no non-finite values. But mean batch Q rises steadily (QRM 4 → 10 → 23 →
  33 at 10/20/50/100%; DRQN plateaus at ~16–27) while evaluation returns stay ~0. QRM has 6.1–6.8
  of its 7 heads valid per transition, as expected: most steps carry no event, so every head
  is valid.

### 3.2 Colour (3M steps, ε 1 → 0.05 over 600k)
- **Random warm-up (20k steps):** 330 episodes per cell, all timeouts, mean return −0.9, 0
  successes. About 22% of episodes get at least one correct wall, which earns nothing under
  sparse reward.
- Step 0: untrained greedy policies push in one direction and hit the same wall about 20 times
  (return −20.5). By the first evaluation (150k) every cell has learned to **avoid walls**:
  return ~0, length 300, progress 0 (it stays ≤ 0.09 for the rest of training).
- Training episodes at ε = 0.05: return −0.8 to −1.7, progress 0.26–0.57. Late success rate
  0.1–1.1%: DQN-TS 1.1%, QRM-Cue 0.6%, DQN 0.2%, DRQN 0.1%.
- DQN-family TD losses are tiny (1e-5 – 1e-3) because there is almost no reward to fit.
  QRM-Cue's mean batch Q climbs in all seeds (0.3 at 10% → 3.3 at 50%). In seed 1 it diverges
  after 2.4M steps (batch mean 30, gradient norm 12 > clip 10, loss ×20).
  `valid_heads_per_transition` = 27.7–27.9 of 40, as designed (30 navigation heads on ~91% of
  steps, 1 head on cue/delay steps). DRQN trains on 353 of 384 possible transitions per update.

### 3.3 Maintenance (2.048M steps, 4,000-step limit)
- Only 521–563 episodes per run, mean length ~3,800 steps, 92–96% timeouts. **Zero training
  successes in 8,078 episodes.** Max training progress per run: 0–2 services.
- Step 0: −20,100 (on the boundary all 4,000 steps). By 100k steps PPO and PPO-RM reach
  −2,000 to −500 and stay there; Recurrent PPO stays at about −3,300.
- PPO extras: approx-KL 0.005–0.02, clip fraction 0.1–0.25. The policy keeps moving, so this is
  not starvation by gradient clipping. **Action std collapses** to 0.06–0.1 by 2M steps in
  PPO seeds 1/3/5, all Recurrent PPO seeds and PPO-RM seeds 3/4, with `ent_coef` = 0.
  Value loss spikes up to 8,000 at −1,000 events. Explained variance: PPO 0.8–0.9,
  Recurrent PPO 0.74–0.999, PPO-RM 0.02–0.32 (poor critic).

## 4. Behaviour of the saved policies

Script [03_behaviour.py](03_behaviour.py) → [out/behaviour_*.json](out/), [out/behaviour_tables.csv](out/behaviour_tables.csv).
Snapshots at 10%, 50% and 100% of the budget × 5 seeds. Episodes: the first 10 periodic-eval
seeds (maintenance 5), each run greedy (the evaluation policy) and exploratory (ε = 0.05, the
training floor; PPO family: stochastic actions). Event traces of seed 1 are stored in the JSON.

### 4.1 Delivery: frozen greedy policies loop
| Final snapshot | greedy return / progress | greedy: exact state repeat (cycle length) | greedy cells visited | ε = 0.05 return / progress | ε = 0.05 causes |
|---|---|---|---|---|---|
| DQN | 0.4 / 0.04 | 100% of episodes, first repeat by step 3 (cycle 2) | 2.9 | −27.7 / 0.20 | lava 44%, timeout 42%, premature goal 12% |
| DRQN | 1.9 / 0.28 | hidden state differs, but positions cycle with period 2 | 10.7 | −28.2 / 0.72 | lava 52%, timeout 30%, 1 success |
| QRM | 0.2 / 0.02 | 100%, step 4 (cycle 2) | 3.7 | −26.0 / 0.36 | lava 42%, timeout 44% |
| DQN-TS | 0.0 / 0.00 | 100%, step 4 (cycle 2) | 3.6 | −22.0 / 0.28 | lava 40%, timeout 52% |

Every greedy policy, including DRQN (which has memory) and DQN-TS (which is given the task
state), finds a pair of cells within a few steps and alternates between them for the remaining
~495 steps. A deterministic memoryless policy in an aliased gridworld loops forever once it
re-enters a state. The agents pick up a package in only 10–58% of greedy episodes. Q at the
start state is 12–28 while the realised discounted greedy return is ≈ 0 (§5.7).

**Frozen snapshot vs training at the same ε** ([out/delivery_probe.json](out/delivery_probe.json),
20 periodic + 20 diagnostic seeds per run, 100 episodes per cell and setting):

| Final snapshot | ε = 0 | ε = 0.05 | ε = 0.2 | ε = 1 | Training, last 5% (ε = 0.05) |
|---|---|---|---|---|---|
| DQN (deliveries / ep, periodic · diagnostic) | 0.09 · 0.03 | 0.28 · 0.30 | 0.26 · 0.39 | 0.15 · 0.24 | **1.02** |
| DQN-TS | 0.07 · 0.06 | 0.27 · 0.32 | 0.41 · 0.34 | 0.15 · 0.24 | **1.05** |
| DRQN | 0.34 · 0.20 | 0.74 · 0.63 | 0.90 · 0.74 | 0.15 · 0.24 | **0.86** |
| QRM | 0.08 · 0.05 | 0.37 · 0.31 | 0.48 · 0.38 | 0.15 · 0.24 | **0.81** |

The seed set makes no difference, so this is not an eval-seed artefact. Adding noise does not
close the gap: a frozen memoryless snapshot at the training ε delivers **3–4× less** than the
training episodes logged with almost the same weights. Both records are internally consistent
(e.g. late training: 43% lava, 23% premature goal, 1% success and 1.0 deliveries → expected
return −22.5 vs −23.4 logged).

**Snapshot-mixture probe** ([07_churn_probe.py](07_churn_probe.py), [out/churn_probe.json](out/churn_probe.json)):
the same rollouts, but the acting network switches every 4 steps among the 700k/800k/900k/1M
snapshots, reproduce the training behaviour. DQN 0.28 → **1.11** deliveries, DQN-TS 0.24 → **1.30**,
QRM 0.32 → **1.01**. Lava goes from 27–32% to 40–46% and timeouts from 52–54% to 26–38%,
matching training (lava 43–50%, timeout 28–33%). **The online learner's deliveries come from
its constantly changing greedy policy ("policy churn": weights update every 4 steps), which
breaks loops that no single frozen policy escapes.** This is a learning-dynamics effect, not a
train/eval pipeline mismatch.

### 4.2 Colour: walk to a wall, then refuse to touch it
Final snapshots, greedy: 94–100% of episodes have **no wall touch**. The agent moves about
5 blocks from the centre (the walls are 6 blocks away), then oscillates. About 47% of
navigation steps are spent on one position and about 16 positions are visited in total. With
ε = 0.05 the same snapshots touch 1.2–1.9 walls per episode (52–80% of episodes touch one) and
make 0.2–0.6 correct walls per episode, with 0 successes in 200 episodes. Q at navigation start
is 0.9–2.8, while the realised return is 0 (oracle discounted value 4.5).

### 4.3 Maintenance: rest on the boundary, or hover
Two behaviours per seed, the same in greedy and stochastic modes (std ≤ 0.1 where saturated):

| Behaviour | Seeds (final snapshot) | Mean action (thrust, torque) | Episode |
|---|---|---|---|
| **Saturated "fall and rest"**: no thrust (brake), full torque, rocket tilted ~90°, lying on the boundary | PPO 1, 3, 5; Recurrent PPO 1, 3, 4, 5; PPO-RM 3, 4 | (−0.7 … −1.4, +1 … +1.8), clipped to the bound | ~15% of steps on the boundary (−5 each), ~99% off-path; return about −3,400; discounted −54 |
| **Hover**: thrust about half the time, no boundary contact | PPO 2, 4; Recurrent PPO 2; PPO-RM 1, 2, 5 | (0.04 … 0.25, ≈ 0) | 68–74% of steps off-path; occasionally a premature service attempt (−10); return −300 to −600; discounted −2 to −7 |

Rollouts of the final snapshots (5 periodic seeds × 5 seeds, greedy / stochastic; [out/behaviour_tables.csv](out/behaviour_tables.csv)):

| Final snapshot | Return (greedy · stoch.) | Boundary steps | Off-path steps | Stage exits / ep | Codes / ep | Service attempts / ep | Thrust > 0 |
|---|---|---|---|---|---|---|---|
| PPO | −2,146 · −2,059 | 9% | 81% | 0.00 · 0.04 | 0.32 | 3.8 · 5.0 | 32% |
| Recurrent PPO | −3,177 · −3,043 | 14% | 94% | 0.00 | 0.28 | 0.0 · 0.5 | 9% |
| PPO-RM | −1,646 · −1,546 | 6% | 73% | 0.08 · 0.20 | 0.32 | 2.8 · 6.7 | 47% |

The rocket **almost never leaves the first stage** (≤ 0.2 stage exits per episode) and collects
the first stage's code in about 30% of episodes. Service attempts happen before readiness
(−10 each). Every episode runs out the 4,000-step clock; none collects, services and exits.

## 5. Specific suspects

Script [04_suspects.py](04_suspects.py) → [out/suspects.json](out/suspects.json). **C** = confirmed defect or
confirmed fact, **H** = hypothesis with supporting evidence, **✓** = checked and clean.

| # | Suspect | Verdict | Evidence |
|---|---|---|---|
| 5.1 | Reward scale in maintenance, no reward normalisation for PPO | **C (fact), H (as cause)** | Rewards span −1,000 … +1,000 with −5/step boundary and −0.1/step off-path. SB3 PPO normalises advantages but not values (no VecNormalize); value loss spikes to 8,000; PPO-RM explained variance 0.02–0.3. The policy still updates (KL 0.005–0.02), so starvation by gradient clipping is not supported. The scale problem is real, but §5.1b makes the discount the dominant issue. |
| 5.1b | **Discount horizon (found here)** | **C** | γ = 0.99 gives an effective horizon of 100 steps = 1.7 simulated seconds. Oracle on 5 periodic seeds: undiscounted 1,072, **discounted −4.9**. First positive reward at step 667 on average (weight 0.0012); completion at step 1,380 (weight 8×10⁻⁶). Discounted values: hovering on-path −0 to −10, hovering off-path ≈ −10, falling to the boundary −54, random −29. At γ = 0.999 the oracle scores +287. Under the trained objective, solving the task is not better than hovering. |
| 5.2 | Sparse terminal rewards with the 20% ε schedule | **C (fact), H (main cause in colour)** | Late replay windows hold successes per 100k: colour DQN 0.7, DRQN 0.3, QRM-Cue 1.9, DQN-TS 4.0, so P(minibatch of 32 contains a success) = 0.01–0.13%. Delivery: 2.6–19.8 successes, but 264–432 +10 deliveries per 100k (8–13% of minibatches see a positive reward). In delivery the reward is not that sparse; the problem is reaching it safely. |
| 5.3 | Does random exploration reach a reward during `learning_starts`? | **C** | Delivery: yes. 19% of warm-up episodes deliver once (+10), but 0 of about 7,100 succeed and the mean return is −49.4. Colour: 0 rewards in 1,650 warm-up episodes (correct walls earn nothing under sparse). Maintenance (no warm-up, PPO): 0 successes in 8,078 training episodes, at most 2 services in any run. |
| 5.4 | Episode limits vs budgets | ✓ / **C (fact)** | Delivery: 6,300–7,000 episodes per run (mean length 140–160). Colour: 10,000 (limit 300, almost all timeouts). **Maintenance: only 521–563 episodes**, mean 3,800 steps; 1,000 PPO rollouts of 2,048 steps, each about half an episode. Few episodes, few outcomes to learn from. |
| 5.5 | Replay sampling and target updates | ✓ | SB3 DQN syncs the target every `target_update_interval` env calls (`_n_calls`, 1 env); custom agents use `step % target_update`. FrameReplay samples uniformly over valid transitions; 29,271 synthetic checks (ring wrap-around, truncation, termination) found 0 errors. SB3 buffers mark timeouts (`TimeLimit.truncated`), so truncations bootstrap. |
| 5.6 | DRQN sequence sampling | ✓ | Synthetic check: sequences never cross episodes; the stored LSTM state h₀ matches the first frame; burn-in positions and anything after an episode end carry zero loss weight; next frames line up; episode-start alignment is respected. Runs: 180–240 of 256 (delivery) and 353 of 384 (colour) trainable positions per update. The hidden state advances on ε-steps (code). |
| 5.7 | QRM counterfactual heads | ✓ (logic), **C (Q-values invalid)** | Plumbing: the actual head is always valid, with reward, next head and terminal equal to the stored env values (512 sampled transitions per env); valid heads per transition 6.8 / 7 (delivery) and 28.1 / 40 (colour), matching the logged 6.1–6.8 and 27.7–27.9. **But** QRM-Cue navigation heads evaluated on navigation frames exceed the +10 bound on any episode in 4/5 seeds: seed 1 (868 at 3M, 3.3% of head-frame pairs > 10), seed 3 (14.9 at 1.5M), seed 4 (47 at 1.5M), seed 5 (68 at 3M). QRM delivery heads reach 46–120 (bound 80, seed 4 above it). The actual-head Q stays ≤ 10 but is still 0.9–6.9 against a realised 0. |
| 5.8 | Evaluation greedy collapse | **C (mechanism)** | §4.1/4.2: 100% of greedy delivery rollouts enter an exact 2-cycle by step 3–5; colour greedy policies oscillate in front of a wall. Training at ε = 0.05 makes 0.8–1.1 deliveries per episode and greedy makes 0.04–0.3. Part of the gap (ε = 0 → 0.05 on a frozen snapshot: 0.05 → 0.3) is the evaluation rule. The larger part (0.3 → 1.0) is policy churn (§4.1). Neither explains the absent success: the online agent itself succeeds in only 1–5% of late episodes. |
| 5.9 | Pipeline bug (rewards/obs not reaching the learner, done handling, eval/train mismatch) | ✓ none found | No-learning plumbing runs with the baseline configs (training seed 401, env seeds 6,000,000+): per-episode sums of buffered rewards equal the env returns, and `terminated`/`truncated` match, for SB3 DQN, DQN-TS, colour DQN, DRQN (both envs), QRM, QRM-Cue, PPO, PPO-RM and Recurrent PPO (truncated episodes differ only by SB3's γ·V(terminal) bootstrap term, as intended). The RM wrapper raises if RM and env rewards ever differ, and 55 runs completed. Train and eval envs come from the same `build_env`. Eval policies call the same networks greedily. The 3–4× delivery gap between training episodes and frozen snapshots at the same ε is the same on eval and diagnostic seeds, and a snapshot mixture reproduces the training numbers (§4.1). It is policy churn, not a mismatch. |
| 5.10 | Maintenance action saturation | **C (fact)** | §4.3. SB3's Gaussian is unbounded and actions are clipped. Means sit beyond ±1 in 9 of 15 final policies, and std has collapsed (`ent_coef` = 0), so exploration has stopped. |

## 6. Ranked failure-mode hypotheses and the cheapest test of each

Ranked within each cell by how much of the ~0% success they plausibly explain. "Cost" is
for the cheapest decisive check. Every proposed experiment uses training seeds 401–499 and
episode seeds 6,000,000+ and is a diagnostic, not part of any results table.

### Delivery (DQN, DRQN, QRM, DQN-TS)
| Rank | Hypothesis | Status | Cheapest diagnostic (confirms / rejects) |
|---|---|---|---|
| 1 | **Search-and-survive exploration problem.** With only 4-neighbour sensing, layouts randomised every episode and lava 5%, ε-greedy reaches the 3-delivery + goal chain rarely (0.7–5% of late episodes) and dies in lava about 45% of the time. The learned value prefers safety; task memory is not the bottleneck (DQN-TS also fails). | H, strongly supported | Train DQN-TS for 300k steps with `num_packages=1` and with `percent_death_squares=0` (≈ 3 min CPU each). Success ≫ 0 confirms that the chain length / hazard exploration is the barrier; still 0 points to the learner. |
| 2 | **No good frozen policy exists in the learned Q-function**: greedy snapshots alternate between two cells; the training-time deliveries come from policy churn, not from a policy the evaluation can recover. | C (mechanism; §4.1 probes) | Done: ε-sweep and snapshot-mixture probes (no training). Remaining free check: the same mixture on the 200 final-eval seeds, to quantify how far "the online agent" would score above the frozen-policy headline. |
| 3 | **Q overestimation / aliasing**: Q(s₀) 12–28 against realised ≈ 0; QRM heads above the 80 bound. | C (fact), cause H | Recompute Q vs Monte-Carlo return over training-like rollouts for Double-DQN targets in a 300k-step run on seed 401 (≈ 3 min). The gap closing → max-operator bias. |
| 4 | DRQN memory unused (it loops like the memoryless agents) | H | Probe the LSTM state for carried / delivered counts with a linear readout on 50 rollouts of the final snapshot (no training). |

### Colour (DQN, DRQN, QRM-Cue, DQN-TS)
| Rank | Hypothesis | Status | Cheapest diagnostic |
|---|---|---|---|
| 1 | **Sparse reward + wrong-wall penalty → wall-avoidance local optimum.** A completion needs 3–5 correct walls with no intermediate reward, while every wrong touch costs −1. Greedy policies stop in front of walls. | C (behaviour), H (cause) | DQN-TS with `reward_density="dense"` (+1 per correct wall) or `sequence_length_range=(1,1)`, 500k steps on one GPU (~10 min). Learning confirms; still 0 points to perception/learner. |
| 2 | **Replay holds almost no success transitions** (0.3–4 per 100k; < 0.13% of minibatches) | C (fact) | Same run, with success transitions counted; or prioritised / successful-episode replay. |
| 3 | **QRM-Cue Q-divergence** across navigation heads (> +10 in 4/5 seeds, up to 868), from heads bootstrapping off each other through counterfactual updates with almost no reward to anchor them | C (fact), cause H | 300k-step QRM-Cue run, counterfactual updates on vs off (actual head only), logging max nav-head Q (GPU ≈ 5 min each). Divergence only with counterfactuals → counterfactual bootstrapping; both → generic Q-learning instability (try Double-Q / lower lr). |
| 4 | Exploration: ε = 0.05 after 600k steps; 300 steps is enough for the oracle (92) but a 13-step straight run is needed to reach any wall | H | Count wall touches per training episode vs ε in the existing CSVs (correct walls already logged as progress): progress falls from 0.6–0.9 at ε ≈ 0.5 to 0.2–0.6 at ε = 0.05. Free; partly done in §3.2. |

### Maintenance (PPO, Recurrent PPO, PPO-RM)
| Rank | Hypothesis | Status | Cheapest diagnostic |
|---|---|---|---|
| 1 | **Discount horizon**: γ = 0.99 at 60 Hz makes the first task reward (~step 670) and completion (~1,400) invisible; the oracle's discounted return (−4.9) is not better than hovering. | **C** (analytic + oracle measurement) | Free re-analysis done (§5.1b). Decisive learning test: PPO-RM with γ = 0.999 (or an action repeat of 8 with γ = 0.99) for 2M steps on seed 401 (~20 min CPU). Services / successes > 0 confirm. |
| 2 | **Penalty dominance and action saturation**: −5/step boundary and −1,000 exits push toward stalling; unbounded Gaussian + clipping + `ent_coef` 0 → std collapse to 0.05–0.1 at the bound. | C (fact), H (cause) | Same run with `ent_coef` 0.01 and reward scaling (VecNormalize, reward only) as a second arm. |
| 3 | **Too few episodes / outcomes**: ~540 episodes per run, 0 successes, ≤ 2 services. | C (fact) | Covered by 1–2: the count of services per 100k steps is the first thing to rise. |
| 4 | PPO-RM critic poor (explained variance 0.02–0.3) and RM features not exploited | H | Probe the value prediction vs return on rollouts of the final PPO-RM snapshot (no training). |
| 5 | Recurrent PPO: LSTM adds optimisation difficulty (all 5 seeds saturate) | H | Second-order; revisit after 1–2. |

## 7. Does anything invalidate the baseline results?

**No.** Reconciliation is complete and no pipeline defect was found (§1, §5.5–5.9). The CSV/JSON
files are a faithful record of protocol v2 with its untuned defaults, and they can be reported
as such: all methods at ~0% success, with a documented train/eval behaviour gap.

Three findings change how the results may be **interpreted**:
1. Maintenance is trained on an objective (γ = 0.99) under which the oracle is not optimal.
   Its results say nothing about memory; they are a negative control for the discount setting.
2. Normalised returns measure penalty avoidance (stall vs die) and must not be used to rank
   memory types.
3. QRM-Cue (and to a lesser degree QRM) Q-values are invalid, above the return bound. Its
   numbers are those of a diverging learner, not a converged explicit-memory agent.

Any change to γ, reward scaling, entropy, reward density or evaluation rule changes semantics
or defaults. Per the protocol it needs a proposal and a new protocol version with its own
change-log entry. Nothing here was changed.

## 8. Diagnostic seeds used

Recorded in `BASELINE_PROTOCOL.md` §11 (2026-10-28): training seed **401** (no-learning plumbing
checks; dummy RNG of loaded agents) and env/episode seeds **6,000,000+** (exploration noise in
rollouts, plumbing envs, delivery probe). The behaviour rollouts, Q-bound and saturation checks
reuse the first 20 periodic-evaluation episode seeds (1,000,000–1,000,019) without any training,
as requested. No reserved tuning, stress or final-test seeds were used. No new training was run;
the only learner code executed collected ≤ 3,000 steps with zero gradient updates.

## 9. Reproduce

All read-only with respect to `run_scripts/baseline/results` and `validation`; outputs go to
`analysis/baseline_v2/out` and `analysis/baseline_v2/figures`. Policy snapshots exist only on
this laptop (gitignored), so steps 3, 4 (A/C/C2/D) and 6 need them.

```
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/01_reconcile.py      # ~1 min
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/02_learning.py       # ~1 min
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/03_behaviour.py --env delivery
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/03_behaviour.py --env colour
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/03_behaviour.py --env maintenance --episodes 5
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/04_suspects.py       # sections A B C C2 D E F
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/06_delivery_probe.py   # ~15 min
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/07_churn_probe.py      # ~5 min
D:\miniconda3\envs\temporal_gym\python.exe -B analysis/baseline_v2/05_summary.py        # tables + summary.json
```
