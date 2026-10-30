# Stress-test protocol v3

Status: **FROZEN (decision D11, accepted 2026-10-30).** The researcher approved P1–P18 in
section 13 exactly as recommended, including the easy level, with two additions recorded in sections
10 and 12. The difficulty scale (sections 1–3) is fixed. No stress result, at any level including the
default, had been seen when it was fixed (S5). Any later change must be recorded as a new dated
decision in [BASELINE_PROTOCOL.md](../baseline/BASELINE_PROTOCOL.md) §11 and here.

Sources: [baseline protocol §6, §12, §13](../baseline/BASELINE_PROTOCOL.md),
[tuning protocol](../tuning/TUNING_PROTOCOL.md),
[selected configs](../tuning/results/v3/full/selected_configs.json) and
[selection report](../tuning/results/v3/full/selection_report.md),
[diagnostics memo](../../analysis/diagnostics_v3/MEMO.md).

Evidence for this draft (read-only, diagnostic seeds only):

| Script | What it measures | Output |
|---|---|---|
| [measure_levels.py](../../analysis/stress_design/measure_levels.py) | oracle success, oracle length distribution, uniform-random and do-nothing references per candidate level, 200 episodes on episode seeds **6,200,000–6,200,199** | `analysis/stress_design/out/levels_*.json` |
| [runtime_model.py](../../analysis/stress_design/runtime_model.py) | full-budget hours per cell from the verified tuning and v2 records (train and evaluation time separated) | `out/runtime_model.json` |
| [machine_plan.py](../../analysis/stress_design/machine_plan.py) | the proposed three-machine queue and its expected hours | `out/machine_plan.json` |
| [thread_bench.py](../../analysis/stress_design/thread_bench.py) | training steps/s at 1 vs 4 threads on the laptop (delivery DQN and DRQN, maintenance PPO and Recurrent PPO), training seed 401, train-env seed 6,100,401 | `out/thread_bench_*.json` |

No stress (201–299 / 4,000,000+) or final-comparison (301–399 / 5,000,000+) seed has been used.

---

## 1. Temporal-delay axis per environment

One primary axis per environment. Every other setting stays at the v3 main condition
(`agents/common/env_factory.py` `BASELINE_ENV_KWARGS`). A level only overrides the axis value and,
where section 3 requires it, `max_steps`.

| Env | Axis (constructor kwarg) | What lengthens | What else changes | Why this axis |
|---|---|---|---|---|
| Colour | `retention_delay_steps` | blank frames between the last cue and navigation; cue-to-wall span grows step for step | episode is `d` steps longer inside the fixed 300-step limit, so `d` fewer navigation steps before timeout; nothing else (same rooms, cues, rewards, observations, RM) | the only pure delay in the gym. Sequence length and repeated colours change how much must be stored, not how long |
| Delivery | `num_packages` (N) | the hidden delivered count must be held over more deliveries; oracle steps from first delivery to the end: 5 → 26 → 40 → 51 (N = 1, 3, 5, 7) | more work per episode (N pickups and deliveries); more objects on the same 10×10 grid; RM size 2N+1 | lengthens the memory span **without making exploration harder**: random-policy progress per package stays flat (fraction 0.045–0.10). `grid_size` lengthens the span too (26 → 46 at grid 16) but cuts random progress fivefold (0.065 → 0.013) and changes lava/damage counts and the x, y range, so exploration and memory are confounded. `carrying_capacity` *shortens* the span (26 → 18 at capacity 3) and changes what is hidden |
| Maintenance | `readiness_delay_range`, with `max_stages = 4` unchanged | stages between collecting a code and the first stage it can be serviced in: readiness target r ∈ [min, max]; code s (one per pre-readiness stage) waits r − s stages. Oracle collection-to-service span: 468 → 667 → 763 → 1,052 steps | **codes issued = r** (one per pre-readiness stage, `codes_per_collection_stage = 1`), so a longer delay also means more codes to collect and service and a longer solution (oracle p95 1,583 → 2,414 steps) | the delay the proposal names. Raising `max_stages` cannot lengthen the delay: r ≤ `max_code_types` = 3 and ≤ `max_active_requests` = 3, and raising `max_code_types` changes the observation size (radar 2K) and the PPO-RM state size. With `max_stages = 4`, r ∈ {1, 2, 3} is the whole range. `max_active_requests` and distractors change load and confusability, not delay |

Against RQ2: colour isolates delay cleanly. Delivery and maintenance cannot separate delay from task
length with the existing environments. The analysis therefore normalises progress by the episode's
required count (section 9) and reports the confound (section 12). A maintenance variant that
decouples delay from the code count would need new environment code and would make the default
level a different mechanism from the harder levels. I considered it and do not recommend it.

## 2. Levels (exact values)

| Env | easy | **default** (= v3 main condition) | hard1 | hard2 | x-axis for slopes |
|---|---|---|---|---|---|
| Delivery | `num_packages=1` | `num_packages=3` | `num_packages=5` | `num_packages=7` | N |
| Colour | `retention_delay_steps=0` | `retention_delay_steps=2` | `retention_delay_steps=24` | `retention_delay_steps=48` | d (steps) |
| Maintenance | `readiness_delay_range=(1,1)` | `readiness_delay_range=(1,2)` | `readiness_delay_range=(2,2)`, `max_steps=5000` | `readiness_delay_range=(3,3)`, `max_steps=5000` | mean r (1, 1.5, 2, 3) |

Level names `easy`, `default`, `hard1`, `hard2` are used in paths, configs and W&B tags.

**Default = main condition, exactly.** The default level applies no override. Its env kwargs are
`env_kwargs(env)` and must equal, value for value, the `env.kwargs` of every tuning v3 config
(`trials_v3.json`). Its config differs from the tuning config only in `phase`, `mode`, `budget`,
`seed`, the evaluation seed blocks and cadence derived from the budget, checkpoint and logging
intervals, and the new `stress` block (level id, plan hash). Prompt 2 adds a test that removes
exactly those fields from both configs and requires the remaining JSON, hashed with the existing
`config_hash`, to be identical.

Why these values:

- **Colour 0, 2, 24, 48.** 0 is the shortest delay possible (navigation starts straight after the
  last cue). 48 is the largest value that keeps oracle p95 ≤ half of the 300-step limit (138 ≤ 150;
  64 gives 154 and breaks the rule). It also keeps the oracle's first wall touch (step 5L + d + 13 ≤
  86 at L = 5) inside the tuned DRQN training window of 96 steps that starts at the episode start,
  so the cue-to-first-wall dependency stays learnable by truncated BPTT. 24 sits halfway between 0
  and 48, so the four points are roughly evenly spread on the d axis, which helps the slope.
- **Delivery 1, 3, 5, 7.** Even spacing (ΔN = 2) around the default. N = 1 is the strongest easier
  level: the count memory reduces to one delivered bit, and random play succeeds in 1.5% of
  episodes (0% at N ≥ 2), so learners see completions during warm-up. That is the purpose of the
  easy level (floor effect, S1). The oracle stays far inside the limit up to N = 7 (p95 74 vs 250).
- **Maintenance (1,1), (1,2), (2,2), (3,3).** This differs from your starting position of (2,3) and
  (3,3) for the harder levels. (2,3) and (3,3) are close together: mean r 2.5 vs 3, oracle p95 2,324
  vs 2,414. They would put two of the four points near the top of the axis. (2,2) and (3,3) give
  mean r = 1, 1.5, 2, 3 and make each harder level a single fixed delay, not a mixture. Both
  options need the raised limit (section 3). Numbers for (2,3) are in the evidence table, so you can
  still choose it.

## 3. Solvability, oracle length and step limits

200 episodes per level on diagnostic episode seeds 6,200,000–6,200,199, scripted privileged
oracle (`temporal_comp_gym/oracles.py`). Rule (v2 §3.1): limit ≥ 2 × oracle p95 length. Your
rule for this phase: keep the limit unless the level breaks that rule, raise it only for that
level, and report the confound.

| Env / level | Value | Oracle success | Oracle length mean / p50 / p95 / max | Limit | 2 × p95 ≤ limit? | Random success / return / progress fraction | Notes |
|---|---|---|---|---|---|---|---|
| delivery easy | N=1 | 200/200 | 16.8 / 16 / 26 / 30 | 500 | yes | 1.5% / −49.5 / 0.045 | 0 unsolvable layouts; resampled 0% |
| delivery default | N=3 | 200/200 | 32.9 / 32.5 / 44 / 66 | 500 | yes | 0% / −49.8 / 0.065 | resampled 0.5% |
| delivery hard1 | N=5 | 200/200 | 45.0 / 44 / 60 / 71 | 500 | yes | 0% / −47.3 / 0.087 | resampled 0.5% |
| delivery hard2 | N=7 | 200/200 | 55.7 / 54 / 74 / 88 | 500 | yes | 0% / −44.4 / 0.102 | resampled 0.5% |
| colour easy | d=0 | 200/200 | 72.2 / 72 / 90 / 90 | 300 | yes | 0% / −0.78 / 0.089 | |
| colour default | d=2 | 200/200 | 74.2 / 74 / 92 / 92 | 300 | yes | 0% / −0.80 / 0.072 | |
| colour hard1 | d=24 | 200/200 | 96.2 / 96 / 114 / 114 | 300 | yes | 0% / −0.69 / 0.076 | |
| colour hard2 | d=48 | 200/200 | 120.2 / 120 / 138 / 138 | 300 | yes | 0% / −0.67 / 0.070 | d=64: p95 154, breaks the rule |
| maintenance easy | (1,1) | 200/200 | 1,263 / 1,271 / 1,583 / 1,854 | 4,000 | yes (3,167) | 0% / −6,155 / 0 | |
| maintenance default | (1,2) | 200/200 | 1,460 / 1,466 / 1,931 / 2,290 | 4,000 | yes (3,862) | 0% / −6,153 / 0 | v2 measured p95 1,859 on its final-eval seeds |
| maintenance hard1 | (2,2) | 200/200 | 1,637 / 1,626 / 2,077 / 2,318 | **5,000** | at 4,000: no (4,154); at 5,000: yes | at 5,000: 0% / −7,583 / 0 (at 4,000: −6,075) | |
| maintenance hard2 | (3,3) | 200/200 | 1,982 / 1,976 / 2,414 / 2,732 | **5,000** | at 4,000: no (4,829); at 5,000: yes | at 5,000: 0% / −7,548 / 0 (at 4,000: −6,057) | |
| (alternative) | (2,3) | 200/200 | 1,842 / 1,827 / 2,324 / 2,596 | 5,000 | at 4,000: no (4,649) | at 5,000: 0% / −7,542 / 0 (at 4,000: −6,060) | not recommended (section 2) |

Oracle success does not depend on the limit: every oracle episode at every level finishes within
4,000 steps (longest 2,732 steps). The limit matters for learners, which mostly time out.

Maintenance delay in steps (oracle, from collecting a code to servicing it, mean / p95): (1,1)
468 / 707; (1,2) 667 / 1,005; (2,2) 763 / 1,079; (2,3) 956 / 1,366; (3,3) 1,052 / 1,397. Oracle
return rises with the code count (+100 per service): 1,032 / 1,080 / 1,121 / 1,167 / 1,210.
Required codes per episode: 1, 1.535 (mixture of r = 1 and 2), 2, 2.535, 3.

**Raised maintenance limit (hard1, hard2 → 5,000 steps).** 5,000 is the smallest round value above
2 × p95 for both harder levels. It is applied to those two levels only. Confounds, all reported:
(a) harder levels allow 25% more steps per episode, so a learner has more time to act, and the
budget holds about 20% fewer training episodes (episodes mostly time out); (b) raw returns are not
comparable across limits (more steps of off-path or boundary penalties are possible). To keep a
like-for-like comparison, every stress evaluation episode also records **progress, return and
success reached within the first 4,000 steps** (section 9.1), and RQ2 reports maintenance both ways.
The alternative, keeping 4,000 at every level, is in the decision table. It breaks the 2× rule
(ratio 1.66 at (3,3)) and adds time pressure that grows with the delay.

Colour keeps 300 at every level, as your rule says. The cost is that navigation time falls from
273 − 5L steps (d = 2) to 252 − 5L (d = 24) and 228 − 5L (d = 48). This is part of the colour
confound list.

## 4. Reward-machine and agent validity at every level

| Env | Level-dependent sizes | Validity condition | Holds? |
|---|---|---|---|
| Delivery | RM live states / QRM heads / DQN-TS features = 2N+1: **3, 7, 11, 15**; DQN-TS input = 8 + (2N+1) = **11, 15, 19, 23**; base obs 8 at every level | RM built from `env.unwrapped.num_packages`, `carrying_capacity` (`DeliveryRM.from_env`); counterfactual validity table derived per N | yes, by construction; tested per level (below) |
| Colour | QRM-Cue heads 2L + V·L = **40** and DQN-TS task-state 40 at every level (L_max = 5, V = 6); image 84×84×3 | `CueRecordingRM.from_env` refuses fixed wall colours, vocabulary < 4 or infinite mode: none of these change. Delay only changes the pre-navigation phase; rooms are re-randomised and the agent re-centred after every correct wall, so the (target, remaining) abstraction still holds. At d = 0 there is no `delay_start` event and the delay heads are unused (RM goes cue → nav) | yes. Note: all delay steps share one head with a black frame; actions have no effect during the delay, so this aliasing biases only delay-state values, never a navigation choice. It is the same abstraction as at the default level |
| Maintenance | base obs 28; PPO-RM features K + S + 1 = 3 + 4 + 1 = **8**, input **36** at every level (K = 3, S = 4 unchanged); Recurrent PPO and PPO 28 | RM tracks held set, stage and ready from events; readiness target is never an RM input | yes |

Runtime guard that already exists: `RewardMachineWrapper(check_reward=True)` (default in
`runner.build_env`) raises on any step where the RM reward differs from the env reward or the RM
terminal differs from `terminated`. A mismatch in production would fail the run, never pass silently.

Tests Prompt 2 adds (`tests/test_stress_levels.py`, diagnostic episode seeds 6,300,000+):

1. `test_level_kwargs_valid_and_default_is_main_condition`: every level builds; default kwargs == `env_kwargs(env)` == tuning `env.kwargs`.
2. `test_rm_reward_and_terminal_equal_env_every_step[env, level, policy]`: oracle and uniform-random rollouts at all 12 levels (delivery/colour 20 episodes, maintenance 3), RM reward == env reward and RM terminal == env terminated on every step. This extends `tests/test_reward_machines.py::test_rm_reward_terminal_and_state_match_the_env`, which covers the default only.
3. `test_rm_state_matches_hidden_state[env, level]`: RM state == debug task state on every step.
4. `test_delivery_counterfactual_validity_per_level[N]`: the validity table matches env dynamics at N = 1 and 7 (extends `test_delivery_counterfactual_validity_matches_env_dynamics`).
5. `test_colour_cue_rm_accepts_levels_and_abstraction_holds[d]`: `from_env` accepts d = 0, 24, 48; the register equals the sequence after cues; at d = 0 navigation starts on the step after the last cue.
6. `test_feature_sizes_match_level`: DQN-TS, QRM head count and PPO-RM input sizes as in the table above.
7. `test_oracle_solves_each_level`: oracle succeeds on 5 seeds per level within the level's limit.

## 5. Training budgets and evaluation

Unchanged from v3 at every level: training budgets delivery **1,000,000**, colour **3,000,000**,
maintenance **2,048,000** env steps; 21 evaluation points (step 0 and every 5%); periodic episodes
50 / 50 / 20; final 200 / 200 / 100; greedy headline; noisy secondary (ε = 0.05 for the DQN family,
stochastic actions for the PPO family) on the same episode seeds; checkpoints every 10%. The selected
`n_steps` (2,048 or 1,024) and `train_freq` (4 or 8) divide every budget, so step counts are exact.

## 6. Carrying tuned hyperparameters to the full budget

Rule: **every hyperparameter is copied verbatim** (exact JSON values) from
`run_scripts/tuning/results/v3/full/selected_configs.json`, `selected[cell].hyperparams`. Nothing
is recomputed or re-sampled. The same values are used at every level.

What the code already does with those values (checked):

- ε decay is a **fraction of the run's own budget**: custom trainers use
  `frac = step / (eps_fraction × total_steps)` (`agents/drqn/drqn_core.py:78`, `agents/qrm/qrm_core.py:67`);
  SB3 DQN uses `exploration_fraction × total_timesteps`. At the full budget, the selected fraction
  therefore decays over 4× as many steps as in tuning. Example: delivery DRQN `eps_fraction = 0.5`
  decayed over 125,000 steps in tuning and decays over 500,000 here.
- `learning_starts` (warm-up) and `target_update` / `target_update_interval` are absolute step
  counts and stay as selected.
- PPO-family learning rates are constants (no schedule), so nothing scales.

This is your default (same fraction, warm-up unchanged), and no code change is needed. Limitation:
tuning compared configurations under a 4× shorter absolute exploration period. The rank order at
25% need not hold at 100%, and a config chosen partly for its exploration schedule may behave
differently. The alternative, keeping tuning's absolute decay length (fraction / 4), would change the
selected values, and I don't recommend it.

## 7. Seeds and episode-seed blocks

| Use | Training seeds | Train-env first reset | Periodic eval episode seeds | Final eval episode seeds |
|---|---|---|---|---|
| default level (RQ1 + first RQ2 point) | **301–305** | 10,000,000 + 1,000 × seed | `final_test` block: **5,000,000 + i** (i < 10,000) | new sub-block `final_test_terminal`: **5,010,000 + i** (i < 990,000) |
| easy, hard1, hard2 | **201–203**, the same three at every level (paired) | 10,000,000 + 1,000 × seed | `stress_eval` block: **4,000,000 + i** (i < 10,000) | new sub-block `stress_terminal`: **4,010,000 + i** (i < 990,000) |
| noisy evaluation | as above | — | same seeds as greedy (existing `rng_offset` 7,000 rule) | same |
| expansion (optional, S3) | 306+ (default), 204+ (others) | — | same blocks | same blocks |

The sub-blocks mirror `tuning_terminal = (3,010,000, 990,000)` in `agents/common/seeding.py`. They are
added to `EVAL_SUBBLOCKS`; no existing base, block or stream index is renumbered. Every level and
algorithm uses the same evaluation seeds, so comparisons are paired by episode seed within a block.
Layouts differ between levels anyway, because the env kwargs differ. Seed permissions: stress
training seeds 201–299 and final-comparison seeds 301–399 are allowed only for phase `stress`, each
only with its own evaluation block (default level ↔ 5,000,000 block; other levels ↔ 4,000,000
block). Baseline and tuning permissions do not widen. Diagnostic seeds 401–499 / 6,000,000+ remain
for probes only.

## 8. Run matrix

| | Delivery (4 algos) | Colour (4 algos) | Maintenance (3 algos) | Runs |
|---|---|---|---|---|
| default × seeds 301–305 | 20 | 20 | 15 | **55** |
| easy × seeds 201–203 | 12 | 12 | 9 | 33 |
| hard1 × seeds 201–203 | 12 | 12 | 9 | 33 |
| hard2 × seeds 201–203 | 12 | 12 | 9 | 33 |
| **Total** | 56 | 56 | 42 | **154** |

11 × 5 + 11 × 3 × 3 = 55 + 99 = **154 runs**, confirmed. Each run has one plan entry with level,
seed, cell, machine role, slot order, resolved config and config hash (Prompt 2).

## 9. Metrics and analysis plan (fixed now)

### 9.1 Per-episode records (stress phase only; baseline and tuning schemas unchanged)

In addition to the v3 fields, every stress training and evaluation episode records:
`required` (delivery N, colour this episode's sequence length, maintenance codes issued this
episode), from which `progress_fraction = progress / required` is computed in the analysis. On
maintenance levels with a raised limit, evaluation episodes also record `progress_at_4000`,
`return_at_4000` and `success_by_4000`. The required count is read from the environment at episode
end for logging only. It is never part of any observation or `info` an agent sees.

### 9.2 Metrics

- **Success** rate (task-defined), **progress** (raw count, comparable with tuning),
  **progress fraction**, **raw return**, and **normalised return** `(R − R_random) / (R_oracle − R_random)`.
  R_random and R_oracle are the per-level means in section 3 (diagnostic seeds, frozen with this
  protocol). I use uniform random as the lower reference, not "do nothing", because delivery and
  colour have no no-op action, and in maintenance the zero action scores −19,729 (mean, (1,2)) against random −6,153.
  A do-nothing reference would inflate maintenance scores. v2 §5 already used random.
  Normalised return is secondary and never evidence of learning on its own (section 12, item 19).
- **AUC**: trapezoid over the 21 evaluation points against env steps, divided by the budget (as
  tuning). **Final**: the final evaluation (200/200/100 episodes) after exactly the budget.
- **Steps to threshold**: the first evaluation step at which a run's greedy progress fraction is
  ≥ τ, τ ∈ {0.05, 0.10, 0.25, 0.50}. A run that never reaches τ is "not reached" (censored). It is
  reported as a count of runs and never filled in with the budget.
- **References** shown with every plot: step-0 (untrained) evaluation of the same run, uniform
  random, oracle. Colour policies scoring below their step-0 value is flagged, not hidden.
- Greedy is the headline. **Noisy is reported next to greedy in every table and figure, never instead of it.**

### 9.3 RQ1: sample efficiency at the default level (seeds 301–305 only)

- Per cell: greedy success AUC, progress AUC (raw and fraction), raw-return AUC, normalised-return
  AUC, final values, steps to threshold; learning curves (mean and per-seed).
- Aggregation: **IQM over training seeds** (scipy `trim_mean(·, 0.25)`, as in rliable) with **95%
  stratified bootstrap CIs** (Agarwal et al. 2021): runs resampled with replacement within each
  cell (stratum = env × algorithm), 10,000 resamples, percentile intervals, generator PCG64 seed 2026.
  Also report the mean and every seed's value.
- Main contrasts, per environment: **explicit vs implicit**: delivery QRM − DRQN, colour
  QRM-Cue − DRQN, maintenance PPO-RM − Recurrent PPO. Each is reported as an IQM difference with a
  bootstrap CI and the probability of improvement P(explicit > implicit) with a CI. Pooled across
  environments, the same contrasts use normalised return and progress fraction, stratified by
  environment.
- Reference points, never pooled with the groups above: memoryless (DQN, PPO) and the task-state
  control (DQN-TS) against the explicit agent. This separates "given task memory" from QRM's
  counterfactual learning (D4). Per N7, PPO-RM is reported as "PPO given the RM's task state".
- Primary RQ1 metric: greedy progress-fraction AUC (success AUC is reported first but is expected at
  zero). Everything else is secondary.

### 9.4 RQ2: degradation with delay (all four levels)

- Primary RQ2 metric: **final greedy progress fraction**. Secondary: final greedy normalised return,
  raw return and success; their AUCs; noisy versions of all of them. Maintenance at the raised
  levels is reported twice: full episode, and within the first 4,000 steps.
- Per cell: value at each level (mean, IQM, 95% bootstrap CI, per-seed points), plotted against the
  axis value (delivery N; colour d; maintenance mean r, with the oracle's mean collection-to-service
  span in steps as a second axis).
- **Normalised drop** relative to the easy level: `drop_l = 1 − M_l / M_easy`, with a bootstrap
  CI. It is computed only when the easy level's 95% CI lower bound is above its floor reference
  (below); otherwise it is reported as "not estimable (easy level at floor)". Easy and harder levels
  share seeds 201–203, so their drops also get a paired per-seed version. The default level (seeds
  301–305) is unpaired.
- **Slope**: OLS of the per-run metric on the axis value over all 14 runs per cell (3 + 5 + 3 + 3),
  with a 95% CI from a bootstrap that resamples runs within each level. Robustness check: slope on
  level index 0–3. A negative slope whose CI excludes 0 is "degradation". A CI containing 0 in a
  cell that is not at floor is "no detectable degradation with 3–5 seeds", never "robust to delay".
- **Floors.** A cell-level is **at floor** when the 95% CI upper bound of its metric does not exceed
  the larger of the uniform-random and step-0 references for that level. It is **at zero** when every
  run's value is exactly 0. A cell at floor at every level is reported as **"floor at all levels:
  degradation not measurable"**. It gets a grey or hatched marker in figures and is listed separately
  in the slope table, with its slope shown but not interpreted. "No degradation" is never claimed
  for a floored cell. Success is reported with exact counts (k of n episodes) because it is expected
  at or near zero.
- Comparison across memory types under RQ2: the slope difference explicit − implicit per
  environment with a bootstrap CI, only where neither cell is at floor at all levels.

### 9.5 What is labelled exploratory

Anything not listed above (per-r breakdowns inside the default maintenance level, behaviour
diagnostics, comparisons with tuning or v2) is labelled exploratory. Tuning and v2 results are
different protocols and are never pooled with stress results.

## 10. Machine plan

Three rented machines, one process each. Colour runs need CUDA (refused otherwise); delivery and
maintenance run on CPU. Every machine therefore needs a GPU, because the colour runs are spread
across all three.

**Expected hours per run** (`out/runtime_model.json`). Evaluation is a fixed cost per run (greedy
plus noisy, mostly timeout-length episodes), so tuning time × 4 overstates it: maintenance evaluates
about 4.2M steps per run against 2.05M training steps. I separated evaluation time (rate measured
from the final evaluation) from training time, scaled only training by 4, and took the larger of that
projection and the v2 RTX 3090 full-run time plus the noisy evaluation share.

| Cell | hours/run | Cell | hours/run | Cell | hours/run |
|---|---|---|---|---|---|
| delivery DQN | 0.18 | colour DQN | 1.16 | maintenance PPO | 0.54 |
| delivery DQN-TS | 0.19 | colour DQN-TS | 1.23 | maintenance Recurrent PPO | 1.28 |
| delivery DRQN | 0.55 | colour DRQN | **3.71** | maintenance PPO-RM | 0.58 |
| delivery QRM | 0.13 | colour QRM-Cue | 1.12 | | |

One set of 11 cells: **10.7 h** (8.9 h on tuning's faster GPUs), of which colour is 68%. Your
estimate of 9–14 h per set is confirmed at the lower end. Total for 154 runs: **151 machine-hours**
(about 125 on faster GPUs). Your upper figure of 195 h came from scaling evaluation by 4 as well.
Maintenance hard levels add about +25% evaluation time.

**Assignment** (`out/machine_plan.json`). A "set" is all 11 cells at one (level, seed). Each machine
runs whole sets in rounds, so a partly finished machine still has every cell at every level it has
reached:

| Round | Machine A | Machine B | Machine C | What is complete after the round |
|---|---|---|---|---|
| 1 | default s301 | default s302 | default s303 | RQ1 with 3 seeds (~11 h) |
| 2 | easy s201 | hard1 s201 | hard2 s201 | one seed at every level: a first RQ2 curve (~22 h) |
| 3 | default s304 | default s305 | easy s202 | RQ1 complete (~32 h) |
| 4 | hard1 s202 | hard2 s202 | easy s203 | |
| 5 | hard1 s203 (10 cells) | hard2 s203 (10 cells) | colour DRQN hard1 s203, colour DRQN hard2 s203 | everything (~50 h) |
| **Expected** | **50.2 h**, 54 runs | **50.4 h**, 54 runs | **50.4 h**, 46 runs | 151 machine-hours |

Within a set the order is delivery (DQN, DQN-TS, DRQN, QRM), maintenance (PPO, Recurrent PPO,
PPO-RM), colour (DQN, DQN-TS, QRM-Cue, DRQN). The longest run goes last, so an interruption costs
the least. Each machine runs colour 32–36 h of its ~50 h. Expect 2–2.5 days of wall-clock per machine
plus setup, and ± 30% for the GPU model (colour DRQN took 2.6 h on an RTX 4070 Ti SUPER and 3.7 h on
an RTX 3090).

**Threads.** The launcher passes `--threads` explicitly to every run and sets `OMP_NUM_THREADS` to the
same value; both are recorded in `metadata.json`. Recommendation: **`--threads 1`** on every
machine.

Laptop measurement (`out/thread_bench_*.json`, one short run each, selected configs, training seed
401, CPU), steps/s at 1 → 4 threads:

| Cell | Steps timed | 1 thread | 4 threads | Change |
|---|---|---|---|---|
| delivery DQN | 30,000 (10,000 warm-up) | 430 | 433 | +1% |
| delivery DRQN | 20,000 (10,000 warm-up) | 393 | 466 | +18% |
| maintenance PPO | 16,384 (8 rollouts) | 1,299 | 1,154 | −11% |
| maintenance Recurrent PPO | 12,288 (12 rollouts) | 546 | 508 | −7% |

These agents are small MLPs limited by the simulator, and extra threads mostly add overhead. Weighted
by hours, 4 threads would save about 1 h on delivery DRQN (14 × 0.55 h) and cost about 2 h on the
PPO family (14 × 1.8 h). Colour runs on the GPU and could not be measured on the laptop (CPU-only
torch). 1 thread also matches the tuning runs whose throughput the hour estimates come from. The
laptop is not the cloud CPU, and each figure comes from a single short run, so ±10% is within noise.
Either value is acceptable. What matters is that it is passed explicitly and recorded. The
thread count is execution provenance, not part of the config hash.

**Replacement machines (D11 addition).** The assignment above fixes the work of each role (A, B, C).
If a machine dies or must be given up, its unfinished sets run on a **replacement machine with the
same role**, started with `--continue-batch` and identical settings (same plan hash, same role, same
`--threads`, same device policy). Verified complete runs are skipped, and the interrupted run is
resumed or restarted as the launcher records it. The replacement, its hardware and the time it took
over are recorded in the run metadata and in the results report. Runs are never moved between roles
ad hoc. Any rebalancing would need a new dated decision and a new plan hash.

## 11. Disk, RAM and what is kept

| Item | Per machine |
|---|---|
| Peak RAM per process (v2 measured) | SB3 colour 6.0 GB (replay stores obs + next obs), colour DRQN 4.1 GB, QRM-Cue 3.8 GB, others < 1.1 GB. **Minimum 16 GB, 32 GB preferred** |
| VRAM | < 2 GB for every agent |
| Restart data (one run at a time, pruned after verification) | colour SB3 replay ≈ 4.2 GB, custom ≈ 2.1 GB, ×2 during the atomic restart swap → ≤ 9 GB |
| Policy snapshots kept on the machine (every 10% + final) | colour DQN/DQN-TS ≈ 0.29 GB per run, DRQN ≈ 0.1 GB, QRM-Cue ≈ 0.07 GB, others < 0.03 GB → about 6 GB per machine |
| CSV/JSON results | < 0.1 GB per machine (tuning: 41 MB for 154 runs) |
| Software (venv, CUDA wheels) and checkout | ≈ 10 GB |
| **Free-disk gate at preflight** | **≥ 35 GB**; rent ≥ 60 GB |

Kept: every CSV/JSON record (collected in zips, as in tuning), and the **final policy of every run**
(downloaded separately to `$HOME\Downloads\stress-models`, outside the checkout, as a tar.gz with a
`.sha256`, about 1.2 GB for all 154 runs). Restart data is deleted only after a run passes
independent verification, as in tuning. Periodic policy snapshots stay on the machine and are
destroyed with it.

## 12. Deviations and limitations

Carried from tuning (reported again with the stress results):

1. Tuning ran with **1 CPU thread**, not the planned 4, because the launcher never passed
   `--threads`. This phase passes it explicitly (recommended value 1, section 10) and records it.
   Thread count affects throughput, not env-step budgets.
2. **Mixed GPUs** in tuning (RTX 4060 Ti, 4070 Ti SUPER, 4080 SUPER, 3090). Wall-clock is not a
   hardware-controlled comparison. The same is likely in this phase. Hardware is recorded per run.
3. **NumPy CPU-feature workaround** (`NPY_DISABLE_CPU_FEATURES`) was needed on tuning machine B,
   because last-digit differences in 3 sampled learning rates broke the plan check. This phase takes
   values verbatim and compares plans by exact JSON. If the workaround is needed again, it is a
   reported deviation.
4. **25% tuning fidelity**: selections can miss late learners and do not show full-budget superiority.
5. **Weak selection**: several finalists differ by less than their seed spread; 10 of 11 cells were
   decided on progress AUC and maintenance Recurrent PPO on return AUC. All selected-candidate
   confirmation runs had zero terminal greedy success.
6. Python 3.12.15 on cloud machines vs the 3.10 reference in `requirements.txt` (pinned packages match).
7. One interrupted tuning run was resumed (episode-boundary restart).

New in this phase:

8. **Easy level added** (S1), a change from D10, which planned harder levels only, to address the
   floor effect. Recorded as part of decision D11 (2026-10-30).
9. **Hyperparameters were tuned at the default level only** and are used unchanged at every level.
   Away from the default they are not tuned, which may favour the default level in the RQ2 curves.
10. **ε decay length** is a fraction of the full budget: 4× longer in absolute steps than in tuning (section 6).
11. **Delivery's axis also lengthens the task** (N deliveries). **Maintenance's axis also adds codes**
    (r codes, one service each) and lengthens the solution 1.5×. Progress fraction and the oracle
    reference reduce this but do not remove it. Only colour is a pure delay.
12. **Maintenance hard levels use a 5,000-step limit**: more time per episode and fewer
    training episodes at those levels. Reported with within-4,000-step metrics.
13. **Colour keeps 300 steps**: navigation time falls by up to 46 steps at d = 48.
14. **Tuned recurrent windows vs delay**: DRQN trains on 96-step (colour) and 32-step (delivery)
    windows with burn-in, chosen at the default level. Longer delays push dependencies towards or
    past those windows. This is part of what implicit memory faces, and it is named as a likely
    mechanism, not fixed.
15. **Unpaired default point**: the default level uses seeds 301–305 and the 5,000,000 evaluation
    block; the other levels use 201–203 and the 4,000,000 block. Slopes are computed unpaired; drops
    relative to easy are also given paired among easy, hard1 and hard2.
16. **3 seeds per non-default level**: CIs will be wide; the IQM of 3 values equals their mean.
17. QRM-Cue's delay-phase head is shared by all delay steps (section 4); value-only aliasing.
18. Expected floors: tuning showed 0% terminal greedy success everywhere, so RQ2 will mostly rest on
    progress and return, with floors reported as in 9.4.
19. **Normalised return can rise without task progress (D11 addition).** The random-policy reference
    (P8) is a weak lower bound. A policy can score well above it while completing nothing. In
    maintenance, for example, hovering away from the boundary avoids most penalties and services no
    code. Normalised return is therefore a **secondary** metric. It is never used on its own to claim
    learning or degradation. Wherever progress is zero, or the cell is at floor on progress (9.4),
    any change in normalised or raw return is reported as a change in penalty avoidance, not as task
    learning.

## 13. Decisions (accepted 2026-10-30 as decision D11)

The researcher accepted P1–P18 exactly as recommended below, together with the two additions in
sections 10 (replacement machines keep their role) and 12 (item 19, normalised return is secondary).

| # | Decision | Recommendation | Alternative |
|---|---|---|---|
| P1 | Delivery axis | **`num_packages`** | `grid_size` (confounds exploration and hazard counts) |
| P2 | Delivery levels | **N = 1, 3, 5, 7** | 2, 3, 4, 5 (narrower; easy level less likely off the floor) |
| P3 | Colour axis and levels | **`retention_delay_steps` = 0, 2, 24, 48** | 0, 2, 8, 32 (log-like spacing, smaller range) |
| P4 | Maintenance axis | **`readiness_delay_range`, `max_stages` 4** | new env option decoupling delay from code count (new code; not recommended) |
| P5 | Maintenance levels | **(1,1), (1,2), (2,2), (3,3)** | your (1,1), (1,2), (2,3), (3,3) |
| P6 | Maintenance hard-level limit | **5,000 steps for hard1/hard2 only**, plus within-4,000 metrics | keep 4,000 at all levels (breaks the 2× rule; adds time pressure) |
| P7 | Colour limit | **keep 300 at all levels** | 300 + (d − 2) to keep navigation time constant (changes the limit at every non-default level) |
| P8 | Return normaliser | **uniform random** (v2 §5) per level, frozen from section 3 | do-nothing (no no-op in delivery/colour; inflates maintenance) |
| P9 | Primary metrics | RQ1: **greedy progress-fraction AUC**; RQ2: **final greedy progress fraction**; success always reported first | success-first (expected at zero) |
| P10 | Floor rule | **CI upper bound ≤ max(random, step-0)** per level; "floor at all levels: degradation not measurable" | zero-only rule (misses below-untrained colour policies) |
| P11 | Thresholds | **τ = 0.05, 0.10, 0.25, 0.50 progress fraction**, censored if never reached | other values |
| P12 | Hyperparameter carry-over | **verbatim, same fraction of the full budget, warm-up absolute** (what the code does) | keep tuning's absolute decay length |
| P13 | Evaluation seed sub-blocks | **`stress_terminal` 4,010,000+ and `final_test_terminal` 5,010,000+**; periodic at 4,000,000+ / 5,000,000+ | — |
| P14 | Per-episode logging (stress only) | **`required`, plus `progress/return/success` within 4,000 steps on raised-limit levels** | none (fractions would be guessed) |
| P15 | Machine plan | **rounds of whole sets as in section 10, ~50 h per machine** | assign by cell (fails "partial machine covers every cell") |
| P16 | Threads | **`--threads 1`** on every machine, passed explicitly and recorded (no measured net gain from 4) | `--threads 4` (the original tuning plan; +18% delivery DRQN, −7 to −11% PPO family) |
| P17 | Disk/RAM gates | **≥ 35 GB free at preflight (rent ≥ 60 GB), ≥ 16 GB RAM (32 preferred)** | — |
| P18 | Kept artefacts | **final policy per run, downloaded outside the checkout; CSV/JSON collected; restart data pruned after verification** | also download the 10% snapshots (~18 GB) |
