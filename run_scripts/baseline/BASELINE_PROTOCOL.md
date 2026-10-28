# Baseline Protocol v2 — Phase 1 Audit and Frozen Protocol

Status: **FROZEN (protocol_version `v2`, frozen 2026-10-26)**. The researcher accepted the
recommended options for decisions **D1–D4** (§9). Nothing here has been implemented yet; Prompt 2
implements it. Any later change to this protocol must be recorded in §11 (change log) with a
reason, and changes that affect results require a new `protocol_version`.

- Checkout audited: `NEW-temporal-composition-gym`, branch `new_experiments`, HEAD `1912633`,
  working tree clean. It is a standalone clone (own `.git`, not a worktree of the main checkout).
- Interpreter used for probes: `D:\miniconda3\envs\temporal_gym\python.exe` (Python 3.10.20,
  torch 2.14.0 CPU-only locally, SB3 2.9.0, sb3-contrib 2.9.0, gymnasium 1.3.0, minigrid 3.1.0,
  pymunk 7.3.0, wandb 0.30.0).
- Probes were short, read-only scripts run from the session scratchpad with the repo root put on
  `sys.path`. Results quoted below as "probe".

---

## 1. Audit findings

### 1.1 Handoff findings, verified against the code

| # | Handoff finding | Status | Evidence |
|---|---|---|---|
| 1 | SB3 checkpoints omit seed/config and auto-resume | **Confirmed** | `train_dqn.py:11-26,74-91` (same in `train_ppo.py`, `train_recurrent_ppo.py`, `ppo_rm/train_ppo_rm.py`): glob `models/{algo}_{env}_step_*_steps.zip`, newest wins, any seed. Custom trainers (`drqn`, `qrm`) add `seed` to the name but no config identity, and checkpoints omit target net, replay, ε schedule state and RNG. |
| 2 | Remediation script omits budgets | **Confirmed** | `rerun_drqn_image_and_ppo_rm_continuous.sh:28-38` passes no timestep flag, so trainer defaults apply (100k). It also `cd`s to `run_scripts/` (one level too shallow after the folder move), so its `agents/...` paths no longer resolve. Pilot step counts are historical, not re-verifiable (results removed). |
| 3 | DRQN: flattened image MLP, episode-count replay, hidden reset on random action, no burn-in, truncation = terminal | **All confirmed** | `drqn/train_drqn.py:105-130` (`Linear(84*84*3, 64)`), `:44-57` (capacity in episodes; worst case 10,000 eps × 1,000 steps × obs+next ≈ **423 GB**), `:203-205` (`hidden = None` on ε action), `:238` (zero state per window), `:213,245` (`done = terminated or truncated` used in target). No learning until 33 *complete* episodes exist (≈33k steps if episodes time out) — consistent with the pilot's missing TD losses. |
| 4 | Colour RM wrapper fires `correct_wall` during memorisation; RM is a counter; per-wall reward under sparse | **Confirmed by probe** | `correct_wall` emitted at steps 4, 9, 14, 19 of the memorisation phase (seed 0, length-5 sequence). `qrm/train_qrm.py:34-51` RM is a saturating counter that never stores colours and pays `reward_correct_wall` regardless of `reward_density`. |
| 5 | Delivery RM = delivered count (already observed); maintenance RM = readiness (already observed) | **Confirmed** | `hazardous_delivery/wrappers.py:62-64` appends `delivered_packages` to the default observation; `ppo_rm/train_ppo_rm.py:117-126` RM state is `readiness`, which is obs index 18. Neither machine retains carried packages or held maintenance codes. |
| 6 | Lava labelled as premature goal | **Confirmed by probe** | Lava step → reward −50, `propositions = {"reached_goal_prematurely"}` (`rm_wrapper.py:47-55`). Numerically harmless at defaults (both −50) but semantically wrong and breaks as soon as penalties differ. |
| 7 | Maintenance: global RNG, score double count, macro waypoint `[0,0]`, dead toggles | **All confirmed** | Probe: same `reset(seed=123)` → different layouts; Gymnasium `check_env` **fails** the determinism assertion. `core.py:440,476,485` do `score += reward` where `reward` already contains earlier penalties in the same step (example: boundary −5 then service +100 → step reward +95, score delta +90). `core.py:535` macro nav hard-coded. `duration_matching_mode`, `distractor_stations` stored but unused; `enforce_ordered_maintenance` (in the proposal) does not exist. |
| 8 | Readiness threshold can be unreachable; horizon may be too short | **Partly confirmed, and worse** | Default `readiness_delay_range=(1,3)` gives targets {1,2} (exclusive upper bound), so readiness can switch on. But there is no validation for ranges ≥ `max_stages`, and see **N4**: perfect completion is structurally impossible in most default episodes. |
| 9 | Tests are visual demos | **Confirmed** | `tests/*.py` open `render_mode='human'` windows with random actions and `sleep`; no assertions; fail headless. |
| 10 | Unequal budgets, mixed return definitions, no common evaluation | **Confirmed** | No trainer runs a frozen-policy evaluation. SB3 logs rolling training means; custom trainers log single training episodes. ε schedules differ (DQN 10–20% vs DRQN/QRM 50%), update frequency differs (DQN every 4 steps, DRQN/QRM every step), PPO-RM uses old tuning winners (`lr 8.09e-4`, `ent 0.0257`, `n_steps 4096`) while PPO/RPPO use defaults with different `n_steps`. |
| 11 | Stress scripts turn failures into zeros | Out of scope, not re-audited in detail | Stress scripts call trainers through `subprocess.run(..., capture_output=True)` with plain `python`. The baseline pipeline must not reuse this pattern. |
| 12 | Tuning picked newest TensorBoard folder | Out of scope, consistent with code | `tune_qrm.py` imports `glob` + `EventAccumulator`. Not reused. |

### 1.2 New findings (not in the handoff)

| ID | Finding | Evidence / impact |
|---|---|---|
| N1 | **The conda env's editable install of the gym points at a folder that no longer exists** (`...\temporal-composition-gym`, now renamed `OLD-`/`NEW-`). | `pip show temporal_composition_gym`; `import temporal_comp_gym` fails from any directory except the repo root. `python agents/train_*.py` puts `agents/` (not the repo root) on `sys.path`, so every trainer currently fails to import. Before the rename it would have silently imported the *main* checkout's env code. |
| N2 | SB3 trainers never upload SB3 metrics to W&B. | `wandb.init` without `sync_tensorboard=True` (`train_dqn.py:40`, `train_ppo.py:44`, `train_recurrent_ppo.py:44`, `train_ppo_rm.py:95`); `WandbCallback` only logs gradients/model. |
| N3 | Dependency spec is unusable as written. | `requirements.txt` is UTF-16; lacks `wandb` and `pytest`; contains a stale `-e git+...@932acb9` line; both `pygame` and `pygame-ce` are installed (same import name, last-installed wins). |
| N4 | **Deferred Maintenance perfect completion is combinatorially possible in only ~42% of default episodes.** | Probe enumerating the default schedule (1 code per stage, uniform type in {1,2,3}, stations only of the stage's own type, duplicates uncollectable, readiness target in {1,2}): 41.9% of 2,000 schedules allow a perfect run. Failure modes: a held duplicate blocks collecting the stage code (→ −1000 at exit); a held code never meets a matching station after readiness. |
| N5 | Maintenance has no internal time limit. | `step()` always returns `truncated=False`. The only horizon is the trainers' `TimeLimit(1000)` = 16.7 simulated seconds for 4 stages with 2-second hovers. Feasibility unmeasured. |
| N6 | Delivery observation details that matter for memory claims. | The "inventory" flag exposes *delivered* count, never *carried* count. Packages and destinations share sensor value 3 (as in the proposal). Walls read as 0 (same as empty floor). Probe: 0.4% of layouts wall off an objective with lava. Env truncates at `max_steps=500`; the trainers' `TimeLimit(1000)` is dead code. |
| N7 | **PPO-RM as designed is exactly "PPO + privileged task-state features".** | With RM reward equal to env reward and RM state appended to the observation, PPO-RM differs from memoryless PPO only by the extra features. This is legitimate but must be stated: in maintenance the explicit advantage *is* the given task memory. QRM additionally gets counterfactual updates. |
| N8 | QRM counterfactual updates are slow and partly invalid. | `train_qrm.py:265-295` loops in Python over states × batch with per-element GPU writes (~256 tiny kernels per update). Labels such as `reached_goal_successfully` encode the env's own counter, so counterfactual RM states receive the actual outcome rather than their own. Hidden env counters mirror the RM state, so some counterfactual transitions would not occur in the counterfactual state (see §4.1). |
| N9 | DRQN imports `distutils` (removed in Python 3.12) and converts nested Python lists to tensors every step. | Breaks on newer cloud images; slow. |
| N10 | Launchers are cwd-dependent and fragile. | No `set -euo pipefail`; bare `python`; relative paths; inconsistent flags (`--env/--timesteps` vs `--env-id/--total-timesteps`); env config via an undocumented `ENV_KWARGS` env var. Successful-completion banners print regardless of failures. |
| N11 | README result locations conflict with the handoff. | README says `wandb_phase_3_baseline_exports/`; handoff says `run_scripts/baseline/results/`. Protocol below uses the handoff path; README to be updated. |
| N12 | Tuning/stress drivers depend on the current trainer CLIs and `rm_wrapper` classes. | `tune_ppo_rm.py:24`, `stress_test_*.py`. They will break when trainers change. They are out of scope; mark them legacy rather than keep wrong semantics alive for compatibility. |

---

## 2. Comparison matrix

| Env (key) | Memoryless (machine A) | Implicit (machine B) | Explicit (machine C) |
|---|---|---|---|
| Hazardous Delivery (`delivery`) | DQN (SB3) | DRQN (custom, corrected) | QRM with carry/deliver RM (custom, corrected) |
| Sequential Colour (`colour`) | DQN (SB3, CNN) | DRQN (custom, CNN+LSTM) | **QRM-Cue** (custom, §4.3) |
| Deferred Maintenance (`maintenance`) | PPO (SB3) | RecurrentPPO (sb3-contrib) | PPO-RM (SB3 PPO + RM state, §4.2) |

5 training seeds × 9 cells = **45 main runs**, 15 per machine. No frame stacking for any agent (it
would give the memoryless group short-term memory).

Task-state controls (D4, accepted): **DQN-TS** on delivery (base obs + delivery RM features) and
**DQN-TS** on colour (frame + φ(u) features injected after the CNN encoder), same defaults as the
DQN cells, 5 seeds each = **10 control runs**. They are labelled `control`, never pooled with the
memoryless group. Maintenance needs no extra control, because PPO-RM already is PPO plus task state
(N7). Total: **55 runs**.

Evaluation-only references per environment (no training, cheap, run once by the analysis step and
by tests): **uniform-random policy** and **scripted privileged oracle** (§5).

---

## 3. Task configurations, contracts and limits

### 3.1 Main ("non-Markovian") condition per environment

| | Delivery | Colour | Maintenance |
|---|---|---|---|
| Base settings | grid 10, 5% lava, 5% damage, 3 packages, capacity 1, 2 heal nodes, random start + goal, no task dependencies, rewards as now | sequence 3–5, delay 2, vocab 6, no consecutive duplicates, random wall colours, sparse reward (+10 completion, −1 wrong wall), no hard death, no noise, finite mode | 4 stages, readiness target ∈ {1,2}, 3 code types, capacity 3, hover 2 s, `local_grid` sensor, gravity 500, rewards as now |
| Observation all agents receive | x, y, 4 sensors, current square, **health**; **delivered count hidden** (**D1**); carried count hidden | 84×84×3 RGB frame | 25-dim vector (kinematics, 3×3 grid, **macro waypoint now implemented**, floor code, readiness, radar); fixed deterministic scaling wrapper |
| Hidden task memory needed | carried status, deliveries so far, where seen packages/destinations are | cue sequence, progress pointer | held code set, stage index |
| Change from current code | `include_inventory_in_state=False` (**D1**); reject-resample layouts with an unreachable objective | events + time limit only | seeded generation; solvable schedule; **distractor stations on** (**D2**); waypoint; time limit |
| Episode limit (single source, internal) | 500 steps (existing) | **300 steps** (optimal ≤ 87 at length 5) | **provisional 3,000 steps** (50 sim-s); final value = ≥ 2× oracle p95 length |

Truncation at the limit sets `truncated=True`, `terminated=False`, cause `timeout`. Trainers stop
adding their own `TimeLimit`.

Not hidden for anyone, documented as a shared confound: **spatial map memory** (where unseen
objects are) in delivery and maintenance. No reward machine retains it, so RM augmentation does
not make those tasks fully observable. Colour navigation is fully visible, so colour is the
cleanest test of sequence memory.

Optional observability ablations (not part of the 45-run matrix): delivery with delivered count
visible (current default) and with health hidden; maintenance without distractors.

### 3.2 Environment info contract (all three envs)

Every `step` and `reset` returns `info` with:

- `events`: list of `(type, arg)` emitted from the transition semantics at the moment they are
  observable (never inferred from reward values). `reset` carries the first cue in colour.
- `is_success` and `termination_cause` on the final step of an episode (also on truncation).
- No navigation target, cue sequence, held-code set or delivered count in `info` during training.
  A `debug` dict with privileged state exists only when `debug_info=True` (off for all training
  and evaluation; used by tests and oracles).

| Env | Events | Termination causes |
|---|---|---|
| Delivery | `package_contact`, `pickup`, `destination_contact`, `delivery`, `goal_contact`, `lava`, `damage`, `heal`, `health_depleted` | `success`, `lava`, `damage_death`, `premature_goal`, `timeout` |
| Colour | `cue(colour_id)` at each cue item onset (incl. reset), `delay_start`, `nav_start`, `wall_touch(colour_id)`, `room_reset` | `success`, `wrong_wall_death` (only if hard death), `timeout` |
| Maintenance | `code_collected(k)`, `readiness_on`, `service_attempt(k)` (hover timer reached), `stage_exit(s)`, `stage_exit_incomplete`, `final_exit`, `boundary_contact`, `off_path` | `success`, `uncollected_codes_at_exit`, `held_codes_at_final_exit`, `timeout` |

### 3.3 Success criteria

- Delivery: goal reached with all packages delivered.
- Colour: full sequence touched in order before timeout.
- Maintenance: final exit with every code collected and serviced (the current "perfect completion").

Secondary per-episode fields: raw return, length, deliveries / correct walls / services completed,
wrong-wall touches, damage taken.

---

## 4. Reward machines (explicit agents)

Common rules:
- The labelling function is the env's `events` stream. RMs never read env internals.
- **No reward shaping.** RM reward must equal env reward on every step of every real trajectory
  (tested). The RM contributes state (and, for QRM, counterfactual updates), not extra reward.
- Terminal = env `terminated` or RM terminal. Truncation bootstraps.

### 4.1 Delivery RM (QRM)

- States `(d, c)`: deliveries so far `d ∈ 0..N`, carried `c ∈ 0..cap`, `d + c ≤ N`, plus terminal.
  At N=3, cap=1: 7 live states.
- Transitions: `pickup` → c+1; `delivery` → d+1, c−1; `goal_contact` → terminal (+50 if d = N,
  else −50); `lava` → terminal (−50); `health_depleted` → terminal (−50, plus the −1 damage);
  `damage` → −1; `delivery` → +10.
- Memory retained beyond the main observation: both `d` (hidden under **D1**) and `c` (always
  hidden). Under the current default observation it would add only `c`.
- **Counterfactual validity.** Hidden env counters mirror the RM state, so the env's response on
  package/destination squares depends on it. QRM updates a counterfactual state `u'` only if the
  env would have produced the same transition in `u'`:
  `pickup` → `c' < cap`; `package_contact` without pickup → `c' = cap`; `delivery` → `c' > 0`;
  `destination_contact` without delivery → `c' = 0`; all other transitions → every `u'`.
  Implemented as a precomputed `(state, event) → (next_state, reward, terminal, valid)` table, so
  updates are vectorised.
- Information advantage given to QRM (documented): the task structure and ground-truth event
  labels. Pickup vs delivery is not distinguishable from one observation when counts are hidden;
  the labels resolve it.

### 4.2 Maintenance RM (PPO-RM)

- State: held set `H ⊆ {1..K}`, stage index `s ∈ 0..S−1`, ready bit (K=3, S=4: 64 live states).
- Transitions: `code_collected(k)` → add k; `readiness_on` → ready; `service_attempt(k)` →
  +100 and remove k if ready and k ∈ H, −10 premature if not ready, −10 wrong if k ∉ H;
  `stage_exit(s)` → s+1; `final_exit` → terminal, +1000 if H empty else −1000;
  `stage_exit_incomplete` → terminal −1000; `boundary_contact` −5; `off_path` −0.1.
- PPO input: base obs + factored RM features (multi-hot H, one-hot s, ready) = 8 extra values.
  Equivalent to a one-hot of the 64 product states.
- Memory retained beyond the observation: held codes and stage index. Readiness is visible already.
- No counterfactual updates (PPO is on-policy). Per N7, report PPO-RM as "PPO given the RM's task
  state".

### 4.3 Explicit colour baseline: QRM-Cue

The colour task's targets change every episode, so a fixed RM over "progress" cannot work. The
proposed baseline is QRM over a **cue-recording reward machine**:

- **RM state** `u = (phase, R, k)`: phase ∈ {cue, delay, nav, done}; register `R` = list of colour
  ids recorded from `cue` events (length ≤ L_max = 5); pointer `k` = walls completed.
- **Transitions**: `cue(c)` appends c to R (once per cue item, so repeated colours are recorded
  when duplicates are enabled); `nav_start` sets phase nav, k = 0; `wall_touch(w)`: if w = R[k]
  then k+1 and reward (+10 when k reaches |R| under sparse; +1 per wall under dense), terminal when
  k = |R|; otherwise −1 (terminal only under hard death).
- **No hidden-answer leakage**: R is filled only from cue events while cues are on screen; nav
  events carry only the touched wall's colour (visible in the frame), never the target; the env's
  `sequence`/`current_seq_idx` are never read. The env's own correct/incorrect verdict is used
  only in tests, to check it matches the RM's.
- **Q-function**: `Q(o, φ(u), a)` with one head per abstract state
  `φ(u) = (target R[k], remaining |R|−k)` during navigation (6 × 5 = 30 heads) plus one head per
  cue/delay step count (6), on a shared NatureCNN encoder. Justification: every room is
  re-randomised independently and the agent re-centred, so the optimal action and value depend on
  the current target and how many walls remain, not on later targets. A test checks this with
  the oracle.
- **Counterfactual updates** (dynamics-consistent only):
  - nav step without a wall touch → all 30 nav heads, next state same (target, remaining);
  - wrong touch of w (agent reset, same room) → heads with target ≠ w, any remaining;
  - correct touch, episode continues → heads with target = w and remaining ≥ 2, next state =
    (actual next target from the new room, remaining − 1);
  - completion → only (w, 1);
  - cue/delay steps → no counterfactuals (cue content is observed).
- **Replay**: transitions (uint8 obs, action, event id, true φ(u), next φ(u), terminal, truncated);
  100k capacity ≈ 2.1 GB. No per-state growth.
- **Relation to QRM** (Toro Icarte et al. 2018): same `(o, u)`-indexed Q-learning with off-policy
  counterfactual updates across RM states. Three documented differences: (1) a per-episode register
  filled from events (a parameterised RM); (2) heads over a provably sufficient abstraction of u;
  (3) counterfactuals restricted to states where the env would have behaved identically. Report it
  as "QRM-Cue (QRM with a cue-recording RM)", never plain QRM.
- **Information advantage** (documented): symbolic colour ids and cue-item boundaries from the
  labelling function (the RM does not have to learn to recognise colours during cues), plus the
  task structure. The policy network still has to map the symbolic target to the wall colour in
  pixels.

---

## 5. Solvability and controls

| Control | Delivery | Colour | Maintenance |
|---|---|---|---|
| Scripted privileged oracle (`debug_info=True`) | BFS planner over the full grid | moves to the wall matching the env's target | path-following PD controller with known schedule |
| Acceptance | 100% success on the 200 final-eval seeds | 100% | ≥ 95% success, p95 length ≤ 50% of the horizon |
| Uniform random | reported (normaliser) | reported | reported |
| Task-state learning control (**D4**, in the run plan) | DQN-TS: DQN + RM-state features | DQN-TS: DQN + φ(u) features | = PPO-RM already |

Normalised score for cross-env plots: `(R − R_random) / (R_oracle − R_random)`.

Maintenance solvability fix (required, not a decision): generate a seeded per-episode schedule in
which codes appear only in pre-readiness stages with distinct types (no duplicate deadlock), every
held type has a matching station in some post-readiness stage, and the final stage can be exited
clean. Validate `readiness_delay_range` against `max_stages` at construction.

---

## 6. Training and evaluation protocol

### 6.1 Budgets (matched within each environment, env interactions only)

| Env | Budget | Note |
|---|---|---|
| Delivery | 1,000,000 | exact (step-based loops) |
| Colour | 3,000,000 | review after measured throughput in Prompt 4; may drop to 2M for all three algorithms together |
| Maintenance | **2,048,000** | = 1,000 rollouts × `n_steps` 2048 for PPO, RPPO and PPO-RM, so SB3 rounding never adds steps |

Eval steps never count toward the budget and are logged separately.

### 6.2 Untuned common defaults (one family per environment; tuning is a later phase)

| | Delivery (DQN / DRQN / QRM) | Colour (DQN / DRQN / QRM-Cue) | Maintenance (PPO / RPPO / PPO-RM) |
|---|---|---|---|
| Encoder | MLP 128-128 (DRQN: 128 → LSTM 128) | NatureCNN 512 → 256 (DRQN: → LSTM 256); RGB kept, /255 | MLP 128-128 (RPPO: LSTM 128) |
| Optimiser | Adam 5e-4, Huber, clip 10, γ 0.99 | Adam 1e-4, Huber, clip 10, γ 0.99 | Adam 3e-4, clip 0.2, n_steps 2048, batch 64, 10 epochs, λ 0.95, ent 0, max_grad 0.5 |
| Replay | 100k transitions | 100k transitions, uint8 | — |
| Learning starts / train freq | 10k / every 4 steps | 20k / every 4 steps | — |
| Target update | 2,000 steps | 10,000 steps | — |
| ε | 1.0 → 0.05 over first 20% | 1.0 → 0.05 over first 20% | — |
| DRQN sequences | 8 chunks × 32 steps, 16-step burn-in from stored state | 4 chunks × 96 steps, 32-step burn-in (chunks starting at episode start use zero state, no burn-in) | RPPO default episode-cut sequences |

The old PPO-RM tuning winners are not used. Differences that cannot be matched (transitions per
update between DQN and DRQN, number of heads) are recorded per run.

DRQN corrections: hidden state updated on every step including ε-random actions; episode-aware
flat uint8 replay (no cross-episode windows, padding masked out of the loss); target uses
`(1 − terminated)` so truncation bootstraps; target network is also run with burn-in.

### 6.3 Seeds

| Purpose | Values |
|---|---|
| Baseline training seeds | **1, 2, 3, 4, 5** (6–10 reserved for expansion, added to all algorithms in an env together) |
| Tuning training seeds | 101–199 |
| Stress training seeds | 201–299 |
| Tuned final-comparison training seeds | 301–399 |
| Periodic-eval episode seeds | 1,000,000 + i |
| Final-eval episode seeds (baseline) | 2,000,000 + i |
| Tuning validation / stress eval / final test episode seeds | 3,000,000+ / 4,000,000+ / 5,000,000+ |
| Training env first-reset seed | 10,000,000 + 1,000 × train_seed; later resets continue the env RNG |

All per-run RNGs (torch, numpy, python, action-space sampling, replay sampling) are derived from
`SeedSequence([train_seed])`. Eval uses a separate env instance and separate streams, so it cannot
advance training randomness. Every algorithm sees the same eval episodes (paired comparisons).

Five seeds is the starting point (Henderson et al. 2018; Agarwal et al. 2021). Expansion rule,
fixed in advance: if the 95% stratified-bootstrap CI of final success rate is wider than ±0.15 for
any cell in an environment, add seeds 6–10 to every algorithm in that environment.

### 6.4 Evaluation

- Cadence: 20 periodic evaluations per run (every 5% of budget) plus one at step 0.
- Episodes: delivery and colour 50 periodic / 200 final; maintenance 20 periodic / 100 final.
- Action selection: greedy (ε = 0, `deterministic=True` for PPO family). Recurrent state reset at
  episode start and carried within the episode. QRM agents track their RM from the eval env's events.
- Eval never writes replay, updates normalisers or touches optimiser state.
- **Headline = final policy after exactly the budget**, evaluated on the final-eval seeds.
  Best-periodic is secondary and labelled as selected-on-eval.
- RQ1 metrics: success rate and raw env return vs env steps; AUC = trapezoid over the 21 periodic
  points, normalised by budget; final success / return. Aggregation with IQM and bootstrap CIs.

### 6.5 Outputs

```
run_scripts/baseline/results/v2/<env_key>/<algo>/seed<seed>__<cfg_hash10>__<utc>-<rand6>/
  config.resolved.json     # everything: env kwargs, model/optimiser/replay, seeds, limits, budget, eval plan
  identity.json            # protocol_version, phase=baseline, env, algo, seed, cfg_hash, run_uid, machine role
  metadata.json            # git rev + dirty diff hash (no secrets), python/pkg versions, hardware, device actually used, W&B id/url
  status.json              # pending | running | failed | complete (+ steps done, error)
  train_episodes.csv       # global_step, return_raw, length, success, cause, [rm_return]
  train_updates.csv        # global_step, n_updates, loss, q_mean, epsilon, lr, grad_norm, replay_size
  eval_periodic.csv        # eval_idx, global_step, episode_seed, return_raw, length, success, cause, progress
  eval_final.csv           # same fields, final-eval seeds
  checkpoints/             # restart checkpoints every 10% + final
  logs/stdout.log, logs/stderr.log
```

A run is `complete` only after the budget is verified and the final evaluation is written.
Missing or failed runs stay `failed`/absent and are never read as zero.

W&B: project `temporal-composition-gym`, group `baseline-v2`, job_type `<env_key>/<algo>`, tags
`[baseline, v2, <machine role>]`. All metrics use `global_step` (env steps) as x-axis:
`train/*` (episode return, length, success, epsilon, loss, q_mean, n_updates, grad_norm),
`eval/return_mean|return_std|success_rate|length_mean|n_episodes|cause/<c>`, `final_eval/*`,
`sys/steps_per_sec|gpu_mem|ram`. Custom loggers write to W&B directly (no TensorBoard dependence).
Pilot runs stay in their old groups.

---

## 7. Resources and three-machine launch plan

| Item | Estimate |
|---|---|
| Image replay, SB3 DQN colour (obs + next stored, required with timeout handling) | 100k × 21,168 B × 2 ≈ **4.2 GB** |
| Image replay, custom DRQN / QRM-Cue (single copy, index-based next obs) | ≈ **2.1 GB** (+ ~0.1 GB hidden states for DRQN) |
| Peak RAM per process | ≤ ~7 GB → machines need ≥ 16 GB (32 GB comfortable) |
| VRAM | < 2 GB for every agent (NatureCNN ≈ 1.7M params; DRQN batch 4 × 128 frames) |
| GPU need | Colour agents benefit from a GPU; delivery and maintenance agents are CPU/simulator-bound |

Machine roles (one process at a time, **seed-major order** so a partial machine still has balanced
coverage):

| Machine | Runs (per seed, then next seed) | Expected critical path |
|---|---|---|
| A — memoryless (+ controls) | DQN-delivery, DQN-colour, PPO-maintenance; then, after all 15 main runs, DQN-TS-delivery and DQN-TS-colour (seed-major) | DQN-colour, DQN-TS-colour |
| B — implicit | DRQN-delivery, DRQN-colour, RPPO-maintenance | DRQN-colour (heaviest of all) |
| C — explicit | QRM-delivery, QRM-Cue-colour, PPO-RM-maintenance | QRM-Cue-colour |

Wall-clock is not yet measured. Order-of-magnitude guess, assuming a mid-range GPU: A 10–20 h
for the main runs plus 6–12 h for the controls,
B 15–35 h, C 12–25 h. Prompt 4 measures steps/sec on each machine and the colour budget is
revisited then. GPU models may differ; that only affects wall-clock, which is reported separately.

---

## 8. Files to change and acceptance criteria

| Correction | Files | Acceptance test |
|---|---|---|
| Import path not dependent on the broken editable install (N1) | `pyproject.toml`/`pytest.ini` (pythonpath), launchers set `PYTHONPATH` to repo root, trainers run as `python -m agents....` | `tests/test_import_source.py`: imported `temporal_comp_gym.__file__` is inside this checkout, from any cwd |
| Delivery events, causes, main condition, solvable layouts | `envs/hazardous_delivery/{core,wrappers,__init__}.py`, `agent_observation_space.txt` | scripted tiny layouts: lava → cause `lava`, goal early → `premature_goal`, damage to 0 → `damage_death`, heal restores; resampling makes 100% of 1,000 seeds BFS-solvable; same seed → identical layout + rollout |
| Colour events, time limit, no leakage | `envs/sequential_colour/{core,wrappers}.py`, docs | cue events only during cues (one per item, incl. reset and duplicates); `wall_touch` only on real touches; nav `info` has no target; timeout at 300; scripted oracle succeeds 100% |
| Maintenance RNG, score, waypoint, schedule, toggles, time limit (finding 7, 8, N4, N5) | `envs/deferred_maintenance/{core,wrappers,objects,__init__}.py`, docs | `check_env` passes; same seed → same layouts and rollout; score delta == step reward on every step; 100% of 1,000 schedules admit a perfect run; unimplemented options raise; waypoint non-zero and points along the path; oracle ≥ 95% success within half the horizon |
| Reward machines + labellers (findings 4–6, N7, N8) | new `agents/reward_machines/{base,delivery,maintenance,colour}.py`; `agents/rm_wrapper.py` retired | RM reward == env reward on every step over 200 random + oracle episodes per env; RM state == hidden env task state on-trajectory; colour register == env sequence after cues, built without reading env internals; counterfactual validity table unit-tested |
| QRM (delivery + Cue) | `agents/qrm/train_qrm.py` | vectorised update matches a slow reference on a fixed batch; invalid counterfactuals receive zero weight; truncation bootstraps |
| DRQN (finding 3, N9) | `agents/drqn/train_drqn.py` | hidden state advances on ε actions; no window crosses an episode; padded steps have zero loss weight; truncation bootstraps; replay RAM within 10% of the estimate; a 2-step memory toy task is learned |
| SB3 trainers + PPO-RM (findings 1, 10, N2) | `agents/train_dqn.py`, `train_ppo.py`, `train_recurrent_ppo.py`, `ppo_rm/train_ppo_rm.py` | fresh run never loads foreign checkpoints (two-seed collision test); budgets exact; eval logged |
| Shared helpers | new `agents/common/{config,seeding,run_io,evaluation,wandb_utils,env_factory,encoders,replay}.py` | config hash stable and order-independent; run dirs never collide; eval does not change training RNG state (test) |
| Launchers (finding 2, N10) | `run_scripts/baseline/run_{memoryless,implicit,explicit}_memory_baselines.sh`; deprecate `rerun_drqn_image_and_ppo_rm_continuous.sh`; new `baseline_plan.json`, `check_manifest.py`, `preflight.py`, `README.md` | `bash -n`; repo root resolved from any cwd; injected failure → nonzero exit and no success banner; manifest check reports missing runs as missing |
| Dependencies (N3) | `requirements.txt` (UTF-8, add `wandb`, `pytest`, drop stale editable line, one pygame), `setup.py` | clean install in a fresh env imports everything |
| Docs (N11, N12) | `README.md`, tuning/stress folders marked legacy | — |

Tests replace `tests/test_*.py` visual demos (moved to `play_human/` or `tests/visual/`, excluded
from pytest).

---

## 9. Decisions (accepted 2026-10-26)

- **D1 — ACCEPTED: Delivery main condition hides the delivered count**
  (`include_inventory_in_state=False`, health visible). Reason: with it visible the only task
  memory is one carried bit. The visible-count setting is kept as an observability ablation and
  must be described as a change from the pilot default.
- **D2 — ACCEPTED: Maintenance main condition has distractor stations on**, together with the
  solvability fix and the implemented macro waypoint. Reason: with distractors off every visible
  station is correct, so held-code memory is barely needed. Distractors-off is kept as an ablation.
  This changes the proposal's default for `distractor_stations` and must be stated in the report.
- **D3 — ACCEPTED: QRM-Cue (§4.3) is the explicit colour baseline**, with its documented
  information advantage (symbolic cue labels and cue-item boundaries). Reported as "QRM-Cue (QRM
  with a cue-recording RM)".
- **D4 — ACCEPTED: task-state controls DQN-TS (delivery, colour) are part of the run plan**
  (10 runs on machine A after its main runs; §2, §7). They separate "being given the task memory"
  from QRM's counterfactual learning.

## 10. Remaining uncertainties

- Maintenance oracle feasibility and final horizon are unknown until the scripted controller exists
  (Prompt 2). If the oracle cannot reach 95%, the stage geometry or horizon must change.
- Untuned defaults may leave some cells at zero success. That is a valid baseline outcome but weak
  evidence for RQ1; diagnostics (updates done, Q-values, ε) will show whether learning happened.
- Colour compute (DRQN especially) and whether 3M steps is enough are unmeasured.
- sb3-contrib RecurrentPPO's timeout bootstrapping must be verified in Prompt 2.
- The QRM-Cue abstraction relies on symmetric rooms; it must be re-validated for stress settings
  that break symmetry (fixed wall colours, vocab < 4).
- Repointing the conda env's editable install to this clone would change the shared environment;
  the plan avoids needing it.

## 11. Change log

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-10-26 | v2 | Protocol frozen; D1–D4 accepted; DQN-TS controls added to the run plan (55 runs total) | Researcher sign-off after Phase 1 audit |
| 2026-10-26 | v2 | Maintenance episode limit set to **4,000** steps (was provisional 3,000) | Rule in §3.1: ≥ 2× oracle p95 length; measured p95 = 1,859 steps on the 200 final-eval seeds |
| 2026-10-26 | v2 | `readiness_delay_range` is inclusive; baseline value written `(1, 2)` | Same target set {1, 2} as the frozen plan; makes min/max explicit and validated |
| 2026-10-26 | v2 | Colour and maintenance step-limit parameter named `max_steps` | `gym.make` consumes `max_episode_steps` and would bypass the env's own limit and `timeout` cause |
| 2026-10-26 | v2 | Maintenance baseline observation is the fixed-scaled 28-value vector (radar as present/dx/dy) | Replaces the −99 sentinel; still identical for PPO, RecurrentPPO and PPO-RM (PPO-RM: 28 + 8) |
| 2026-10-26 | v2 | DRQN colour: half of sampled sequences aligned to the episode start (`episode_start_fraction=0.5`); gradients also flow through burn-in frames | Trains the cue-to-wall dependencies with exact zero initial state; loss still only on trained positions |
| 2026-10-26 | v2 | Maintenance stages always contain one station per station type; codes appear one per pre-readiness stage with distinct types | Implements the §5 solvability fix and D2 without letting the layout reveal held codes |
| 2026-10-26 | v2 | W&B project defaults to **`new_experiments`** (entity optional), recorded in each run's resolved config; the pilot project `temporal-composition-gym` is refused. Authentication only via `wandb login` on each machine | Updated Prompt 3 instructions; supersedes the project name in §6.5 |
| 2026-10-26 | v2 | Output layout gains a mode level: `results/v2/<full|smoke>/<env>/<algo>/seed<s>__<hash10>__<uid>/`, plus `eval_points.csv`, `resources.csv`, `final_summary.json`; batch logs under `results/v2/<mode>/_batches/` | Keeps smoke outputs physically apart from production; per-point summaries for analysis |
| 2026-10-26 | v2 | Device policy: colour runs on CUDA (fail rather than fall back), delivery and maintenance on CPU; device recorded per run and excluded from the config hash | Image runs need a GPU; the others are simulator-bound |
| 2026-10-26 | v2 | Restart checkpoints every 10% (SB3 at rollout boundaries); resume = episode-boundary restart (see run_scripts/baseline/README.md) | Exact continuation of physics/episode state is not claimed |
| 2026-10-26 | v2 | Smoke mode (labelled, `results/v2/smoke`, W&B group `baseline-v2-smoke`): budgets 4,000 / 3,000 / 4,096; 2 eval points; maintenance limit 600 steps; shortened warm-up/schedules | Pipeline checks only; never part of the manifest |
| 2026-10-28 | v2 | Diagnostic seed ranges reserved and first used: training seeds **401–499**, env/episode seeds **6,000,000+** (exploration noise in behaviour rollouts, no-learning plumbing checks in `analysis/baseline_v2/`) | Baseline analysis (Prompt 1) needs seeds outside the baseline, tuning, stress and final-test ranges; no results or protocol settings change |
| 2026-10-28 | v3 | Protocol v3 defined (decisions D5–D9, section 12): maintenance γ = 0.999; secondary noisy evaluation; extra logging. Selected with `--protocol v3`; v2 stays the default and its configs and hashes are unchanged (`baseline_plan.json` check passes, manifest 55/55) | Evidence in `analysis/baseline_v2/REPORT.md` and `analysis/diagnostics_v3/MEMO.md` |
| 2026-10-28 | v2/v3 | Extra per-episode columns `stage_exits` (maintenance) and `wall_touches` (colour) in training and evaluation CSVs, and `stage_exits_mean` / `wall_touches_mean` in `eval_points*.csv`; empty where not applicable. `final_summary.json` gains `protocol_version` and `final_eval_noisy` (null for v2) | Logging only; collected v2 results are unaffected |

## 12. Protocol v3 (decisions accepted 2026-10-28)

Protocol v2 is frozen: its 55 baseline runs (`results/v2/full`, commit `10334db`) are the reported
record of the untuned default protocol. They are valid but all cells sit at about 0% success. The
analysis (`analysis/baseline_v2/REPORT.md`) found no pipeline defect, and the diagnostics
(`analysis/diagnostics_v3/MEMO.md`, training seed 401 and episode seeds 6,000,000+) traced the floor
to causes unrelated to memory. Protocol v3 makes the minimal evidence-based corrections, each
applied identically to every algorithm in the affected environment. All later phases (tuning, the
tuned final comparison, stress testing) use v3. v2 and v3 results are reported separately.

- **D5 — ACCEPTED: the greedy evaluation stays the headline metric.** v3 adds a pre-registered
  *secondary* evaluation with training-time action noise, reported next to it (never instead of
  it): ε = 0.05 for DQN, DQN-TS, DRQN, QRM and QRM-Cue (the greedy Q-values are still computed every
  step, so recurrent state advances), and stochastic action sampling for PPO, Recurrent PPO and
  PPO-RM. It uses the same episode seeds as the headline. Its randomness comes from a per-episode
  generator seeded by the episode seed, with torch's RNG forked and restored, so it is reproducible,
  paired across algorithms, and isolated from training. Files: `eval_points_noisy.csv`,
  `eval_periodic_noisy.csv`, `eval_final_noisy.csv`; W&B prefixes `eval_noisy/` and
  `final_eval_noisy/`. Reason: frozen greedy policies loop in 100% of delivery rollouts, so the greedy
  headline alone hides what the learners can do. Changing the headline after seeing results would not
  be defensible.
- **D6 — ACCEPTED: maintenance γ = 0.999 for PPO, Recurrent PPO and PPO-RM.** At 60 control steps per
  simulated second, γ = 0.99 looks about 1.7 s ahead. Under it, even the oracle's discounted return
  (−4.9) is no better than hovering; at γ = 0.999 it is +287. The diagnostic probe showed the expected
  change in behaviour (no resting on the boundary, more stage exits) but no services yet at a quarter of
  the budget. This is a correction of the objective, not tuning.
- **D7 — ACCEPTED: colour keeps the sparse reward.** Dense reward was considered but not tested (no
  GPU probe was run, for time and cost). The write-up states this as a limitation.
- **D8 — ACCEPTED: no reward shaping and no action repeat in maintenance.** If no method services
  codes within the budget under v3, that is reported as a result.
- **D9 — ACCEPTED: budgets unchanged** (delivery 1,000,000; colour 3,000,000; maintenance 2,048,000), so
  v2 and v3 are directly comparable.
- **Extra logging (v3):** SB3 DQN and DQN-TS log `q_mean` and `grad_norm` per update round
  (`agents/common/sb3_extras.py`; the update itself is unchanged, checked by test). Every protocol logs
  per-episode `stage_exits` (maintenance) and `wall_touches` (colour).
- **Unchanged:** tasks and env settings (no delivery changes; the delivery probe showed that chain
  length and lava are not the barrier), observations, reward machines, hyperparameters other than
  maintenance γ, seeds and seed ranges, evaluation cadence and episode counts, and the output contract
  (`PRODUCTION_LAUNCH_v2.md` section 9) plus the files above.
- **Implementation:** `agents/common/baseline_plan.py` (`resolve_config(..., protocol_version="v3")`,
  `PROTOCOLS`), runner flag `--protocol v3`, results under `results/v3/...`, W&B group `<phase>-v3`.
  Tests: `tests/test_protocol_v3.py`.
- **Tuning (next phase):** starts from v3. Tuning training seeds 101–199, validation episode seeds
  3,000,000+; tuned final comparison training seeds 301–399, episode seeds 5,000,000+. Selection uses the
  greedy headline (D5); the noisy evaluation is reported but not optimised.
