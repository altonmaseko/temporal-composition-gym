# Tuning protocol v3 — proposed design (T1)

Status: decision-ready proposal; no search or training authorised by this document. Sources: [baseline protocol §12](../baseline/BASELINE_PROTOCOL.md), [collection contract §9](../baseline/PRODUCTION_LAUNCH_v2.md), [v2 analysis](../../analysis/baseline_v2/REPORT.md), and [diagnostic memo](../../analysis/diagnostics_v3/MEMO.md). Freeze this design and its generated plan before production tuning.

## 1. Scope, fidelity and selection

Cover all **11 cells**: delivery and colour each use DQN, DQN-TS, DRQN and QRM (QRM-Cue for colour); maintenance uses PPO, Recurrent PPO and PPO-RM. Preserve v3 tasks, observations, reward machines, sparse colour reward, discount factors (0.99 discrete; 0.999 maintenance), greedy headline and fixed secondary evaluation. No reward scaling/shaping, action repeat, algorithm substitutions or architecture changes. In particular, keep Recurrent PPO LSTM size 128: although suggested as tunable, varying it conflicts with the fixed-architecture requirement and increases compute uncertainty. DRQN encoder/LSTM sizes also remain fixed.

Per cell, use **8 distinct candidates**: candidate 00 is the v3 default anchor; 7 are seeded random draws. Generate once using NumPy PCG64 seed 101, stable cell order and separately identified child streams. Discrete choices are uniform; log-uniform means uniform in log space. Reject duplicate configurations before training. Save sampler/version, spaces, complete resolved configurations, hashes and execution slots in immutable `tuning_plan.json`; never regenerate on resume. This anchor costs one search slot equally in every cell.

Every candidate trains fresh on seed **101**. Confirm the top two by fresh runs on **102, 103, 104**, at the same fidelity. Thus 88 screening + 66 confirmation = **154 runs** (14/cell). Shared seeds across candidates improve pairing; they do not count as independent replications. Derive training environment reset seeds as `10,000,000 + 1,000 * training_seed`, with independent named learner/replay/exploration streams.

| Environment | Training interactions/run | Fraction of v3 budget | Periodic evaluation interval |
|---|---:|---:|---:|
| Delivery | 250,000 | 25% of 1,000,000 | 12,500 |
| Colour | 750,000 | 25% of 3,000,000 | 37,500 |
| Maintenance | 512,000 | 25% of 2,048,000 | 25,600 |

These budgets divide every allowed train frequency/PPO rollout length; evaluation occurs at step 0 and every 5% (21 points), independently of update boundaries. Keep episode limits 500/300/4,000 and v3 periodic counts 50/50/20. Use validation seeds `3,000,000+i` at every periodic point; terminal validation uses `3,010,000+i`, 200/200/100 episodes. Both greedy and noisy evaluations use the same seeds, counts and isolated per-episode RNG rules. Noisy epsilon stays **0.05**, even when training epsilon is tuned. Terminal validation is reported, not used to select a checkpoint or configuration. Evaluation interactions are recorded separately from training budgets. No early pruning.

Rank verified screening runs by the descending tuple **(greedy success AUC, greedy progress AUC, greedy raw-return AUC)**, each computed by trapezoidal integration over actual environment steps divided by the trial budget. Progress means task-defined deliveries, correct walls or successful services; it distinguishes partial learning when success is zero without introducing a shaped reward. Use full-precision values, exact ties, then ascending candidate ID. Rank the two finalists by the componentwise mean of their AUC tuples over **confirmation seeds 102–104 only**, using the same lexicographic rule; report each seed and variability. Excluding screening seed 101 limits selection optimism. Never optimise noisy scores. All-zero success remains a null success result even if tie-breaks select a configuration.

Training epsilon decay is a fraction of the **reduced** budget; warm-up stays at v3's absolute step count. This deliberately differs from truncating a full-budget run and must be reported. The 25% search may miss late learners; it cannot establish full-budget superiority. Only tuning seeds 101–104 and validation blocks above are used. Reserve stress 201–299/4,000,000+, final comparison 301–399/5,000,000+, diagnostics 401–499/6,000,000+; exclude baseline seeds entirely.

## 2. Fixed search spaces

All unlisted parameters retain v3 defaults, including replay capacity, gradient clipping, optimiser, gamma, network sizes and DRQN episode-start sampling fraction. Candidate 00 is exempt from sampled ranges only where explicitly needed to preserve the default. Equal interaction budgets allow different update counts; log both updates and runtime.

| Learner parameters | Distribution / allowed values |
|---|---|
| All DQN-family learning rates | Log-uniform [1e-5, 1e-3] delivery; [1e-5, 3e-4] colour |
| Exploration fraction; final epsilon | {0.2, 0.5, 0.8}; {0.05, 0.1, 0.2}; initial epsilon fixed at 1 |
| Target update interval (environment steps) | Delivery {1,000, 2,000, 5,000}; colour {2,000, 5,000, 10,000} |
| Train frequency; DQN/DQN-TS/QRM/QRM-Cue batch | {4, 8}; {32, 64} transitions; one gradient update per training opportunity |
| DRQN (train length, burn-in) | Delivery {(32,16), (64,16)}; colour {(64,32), (96,32)} |
| DRQN batch | Delivery {4,8} sequences; colour {2,4} sequences |
| All PPO-family learning rate; entropy coefficient | Log-uniform [1e-5, 1e-3]; {0, 0.001, 0.01, 0.03} |
| PPO rollout length; batch size | n_steps {1,024, 2,048}; batch {64,128}, each dividing the rollout |
| PPO epochs; GAE lambda; clip range | {5,10}; {0.90,0.95,0.98}; {0.1,0.2,0.3} |

Motivation: exploration addresses avoidance/rare successes; learning rates, target intervals and update density probe Q inflation; DRQN windows probe temporal credit. PPO entropy addresses action collapse; rollout/advantage settings probe delayed credit and critic optimisation. Diagnostic probes were limited and do not establish causal fixes.

## 3. Machines and spending

Use three user-started RTX 3090 machines, **one process each**, four CPU threads; colour requires CUDA, other tasks use CPU. Balance by cost rather than memory family. Freeze each cell's 14 run slots (screening 00–07, then finalist rank/seed slots) and assign successive slots according to these counts; finalist identities fill existing slots after reconciliation.

| Machine | Allocation | Linear reference hours | Planning billed hours / cost |
|---|---|---:|---:|
| A | 7 colour DRQN; 14 maintenance Recurrent PPO; 6 delivery DRQN | 10.67 | 21–32 / $6.40–9.60 |
| B | 7 colour DRQN; 8 delivery DRQN; all 42 other delivery runs; 14 maintenance PPO; 3 maintenance PPO-RM | 10.64 | 21–32 / $6.40–9.60 |
| C | all 42 other colour runs; 11 maintenance PPO-RM | 10.61 | 21–32 / $6.40–9.60 |

Reference = run counts × 0.25 × full-run hours in REPORT §1 (delivery 0.16/0.51/0.11/0.16; colour 0.98/3.66/0.72/1.08; maintenance 0.42/1.00/0.32). Total linear reference **31.92 machine-hours**. It underestimates fixed evaluation costs: episode counts are retained and noisy evaluation doubles evaluation work. Use a provisional 2–3× allowance: **64–96 rented GPU-machine-hours, about $19–29 at $0.30/hour**, including CPU work on rented machines; GPU-active hours are lower. These are planning estimates, not measured v3 runtimes or price quotes; storage/transfer/idle fees are extra. Recommend a **100 billed-hour / $30 compute cap**. Preflight and first completed trials refine projections before further launches; if the cap is threatened, pause for a scope decision rather than silently shortening runs or dropping slow cells. Schedule confirmation only after all eight candidates in that cell are reconciled; collection between machines may add idle time.

## 4. Records, failures and implementation gates

Mirror the baseline guide's setup → preflight → launch → restart → collect flow. Use phase `tuning`, project `new_experiments`, group `tuning-v3` (`tuning-v3-smoke` for smoke); laptop W&B disabled/offline. Production machines and authentication remain user-managed. Add tuning-specific configuration/seed validation and protocol-aware orchestration without widening baseline seed permissions or changing baseline defaults.

Use compact paths, e.g. `run_scripts/tuning/results/v3/full/delivery/dqn_ts/t00/s101/a00/`. Full hashes and identities live in JSON; attempts never overwrite each other. At the stated 92-character checkout root this example plus `config.resolved.json` is about 173 characters. Enforce **<240 characters for every generated, temporary and extracted path**, including batches/checkpoints. Full mode here means the registered tuning fidelity, not the full baseline budget. Smoke is excluded from selection and uses tuning seeds; its smaller execution settings are labelled explicitly.

Retain the §9 CSV/JSON contract, v3 noisy files/counters, plan hash, candidate ID, stage, seed, attempt and AUC tuple. Objectives come only from the identified run's validated CSVs; verify hashes, budgets, updates and every evaluation point/episode. Failed/missing/interrupted/nonfinite results are unavailable, never zero. Retry infrastructure failures with identical settings; numerical failures remain failed candidates. Resolve duplicates explicitly. Selection requires eight resolved screening attempts, two valid finalists and six valid confirmation runs; otherwise label the cell incomplete. Preserve failure evidence; retries count against the cap.

Keep latest restart state and final policy per run on-machine; never reuse baseline checkpoints. Collect CSV/JSON only; optionally bundle finalists' final policies separately after checking size. `selected_configs.json` records each cell's selected learner parameters, v3 provenance, plan/config hashes, candidate ID, screening/confirmation tuples, per-seed evidence paths, failures, tie-break reason and zero-success flag. Reduced budgets/seeds/evaluation settings remain provenance. Final-comparison execution and stress testing are outside scope.

Before implementation changes, capture regression fixtures for **all v2/v3 baseline configurations and hashes** (full/smoke, seeds 1–10); add tests preserving them, rejecting seed leakage, and verifying deterministic plans, objective ranking, failure handling and path lengths. Require `python -B -m agents.common.baseline_plan write --check` and baseline manifest **55/55**, without writing results/v2 or validation/. On Windows use only `D:\miniconda3\envs\temporal_gym\python.exe`, `-B`, and pytest `-p no:cacheprovider`. T1 creates only this document; tests and implementation belong to later prompts.

## Researcher decisions before freezing

- **Trial count:** recommend 8 candidates/cell including the default, plus top-two confirmation on three new seeds (154 runs).
- **Fidelity:** recommend 25%; accept the explicit limitation on late learning rather than increasing cost now.
- **Spending:** recommend a 100 billed-hour / approximately $30 compute cap, with a pause before exceeding it.
- **Over-budget scope:** recommend retaining all 11 cells. Any cell removal or common trial-count/fidelity reduction requires your decision and a revised plan before affected runs; incomplete searches remain labelled incomplete.
