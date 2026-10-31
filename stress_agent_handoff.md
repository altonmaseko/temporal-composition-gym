# Stress-test agent handoff (protocol v3, final GPU phase)

Send the shared context first, then the prompts in order to the **same agent**. Each prompt ends at a stop point. Do not send the next prompt until you have checked and accepted the agent's result. Renting machines and starting production runs stay your job.

| Order | Prompt | Ends with | Rented machines? |
|---|---|---|---|
| 0 | Shared context | Agent confirms it understands, no edits | No |
| 1 | Design the stress protocol | `STRESS_PROTOCOL.md` draft and decision table, **stop for approval** | No |
| 2 | Freeze the plan and implement configs | Frozen plan file, tests passing | No |
| 3 | Launchers, collection and smoke rehearsal | Offline smoke rehearsal passing | No |
| 4 | Production launch guide | Per-machine `PRODUCTION_LAUNCH.md` | No (you rent afterwards) |
| 5 | Verify results and answer RQ1 and RQ2 | Analysis report and figures | No |
| 6 | Optional seed expansion | Only if time and money allow | Optional |

## Prompt 0: shared context

```text
I am continuing the Temporal Composition Gym experiments on the branch new_experiments (it is the repository's default branch). Read AGENTS.md first. On my Windows laptop every Python command must use D:\miniconda3\envs\temporal_gym\python.exe, run with -B, and pytest with -p no:cacheprovider. Install packages only with that interpreter's -m pip. Cloud Linux scripts must use the explicitly provisioned cloud Python (the tuning launchers use ~/tcg/bin/python), never my Windows path.

Do not commit or push unless I ask. When I ask, write descriptive commit messages, never mention Claude or AI, and add no co-author or attribution trailer. Never run git checkout, stash, reset or clean on files you did not create.

My research questions:
1. RQ1: How do explicit automaton-based reward representations compare with implicit memory-augmented networks in sample efficiency?
2. RQ2: How does performance degrade as temporal delay increases?

Where the project stands:
- Protocol v2 baselines (55 runs, untuned defaults) are done and reported separately: run_scripts/baseline/BASELINE_PROTOCOL.md, analysis/baseline_v2/REPORT.md.
- Protocol v3 (section 12 of BASELINE_PROTOCOL.md; diagnostics in analysis/diagnostics_v3/MEMO.md) fixed maintenance gamma = 0.999 and added a secondary noisy evaluation. Greedy evaluation is the headline.
- Tuning v3 is complete and published (commit 9a2103c): run_scripts/tuning/TUNING_PROTOCOL.md, run_scripts/tuning/results/v3/full/selected_configs.json, selection_report.md, and the collection contract in run_scripts/tuning/PRODUCTION_LAUNCH.md Part 12. 154/154 runs verified. Tuning ran at 25% of the full budgets.
- Decision D10 (BASELINE_PROTOCOL.md section 13) defines this phase: stress tests answer RQ2, and their default-difficulty level, run with the tuned configs at the full v3 budget on the final-comparison seeds, is also the tuned RQ1 comparison. There is no separate final-comparison phase.

The 11 cells (algorithm x environment), each using its selected tuned config:
- Delivery: DQN (memoryless), DQN-TS (given task state, control), DRQN (implicit memory), QRM (explicit reward machine).
- Colour: DQN, DQN-TS, DRQN, QRM-Cue (explicit; documented information advantage).
- Maintenance: PPO (memoryless), Recurrent PPO (implicit), PPO-RM (explicit).

What the tuning results show (read selection_report.md and selected_configs.json yourself to confirm):
- Terminal greedy success is 0% in all 154 runs. Noisy evaluation reached 0.5-2% in 21 runs. Success alone cannot show degradation, so progress (deliveries, correct walls, services) and raw return must carry RQ2, with success still reported.
- Delivery DRQN is the only cell with clear learning (greedy progress AUC about 0.28, still rising at the end of the 25% budget). QRM is near 0.007.
- Colour: trained greedy policies score below the untrained step-0 policy on progress. They learn to avoid walls and wait for the 300-step timeout (sparse +10 reward vs -1 per wrong wall).
- Maintenance: almost all episodes time out. Recurrent PPO has zero progress in every run.
- Several finalists differ by less than their seed spread, so the selections are weakly identified.

Accepted decisions for this phase (I have signed these off):
- S1. Levels: each environment gets the default level plus 1 easier and 2 harder levels along one temporal-delay axis, so 4 points per curve. The easier level is a change from D10, which planned harder levels only. It is there because of the floor effect and must be recorded as a dated protocol decision.
- S2. Seeds: default level 5 training seeds (301-305) with final-comparison episode seeds (5,000,000+). Each non-default level 3 training seeds (201-203, the same seeds at every level so levels are paired) with stress episode seeds (4,000,000+). The default level is run once and never re-run on stress seeds.
- S3. Seed expansion (section 6.3 rule) is optional and only if time allows. It cannot trigger while every cell is at 0% success.
- S4. This is the last GPU rental phase. Everything must work the first time: rehearse locally before any rental.
- S5. The difficulty scale must be fixed and frozen before any stress result is seen, including the default level. Smoke runs are pipeline checks only.

Rough cost (verify it): one set of all 11 cells at full budget is about 9-14 machine-hours (v2 baseline measured about 9 h on RTX 3090s; tuning times x4 give about 14 h). Colour is about 70% of that; colour DRQN is the longest single run (3-3.7 h). Default level 5 seeds plus 3 levels x 3 seeds is about 14 sets: 125-195 machine-hours, roughly 2-3 days on three machines.

Lessons from the tuning rollout to keep:
- Reuse the tuning machinery and patterns (run_scripts/tuning: generate/plan freeze with hash, launch.py, run_trial.py, lib_launcher.sh, preflight.py, check_manifest.py, collect.py, merge_bundles.ps1, rehearse.ps1, --continue-batch restarts, pruning restart data only after verification). Keep the baseline and tuning code paths, plans, hashes and seed permissions unchanged; their regression checks must still pass.
- Missing, failed, interrupted or nonfinite results are unavailable, never zero.
- Do not rebuild anything from floating-point random draws at runtime. Tuning machine B failed preflight because NumPy's CPU-specific exp/log changed the last digit of 3 learning rates. Take hyperparameters verbatim from selected_configs.json and compare frozen plans by exact JSON values. (Workaround that was used on B: NPY_DISABLE_CPU_FEATURES with the AVX512 feature list; it must be reported as a deviation if needed again.)
- Tuning runs used 1 CPU thread because the launcher never passed --threads (the protocol planned 4). This phase must pass --threads explicitly and record it.
- Collected zips contain CSV/JSON only. Final policies are downloaded separately to a folder outside the repository. Every Windows path stays under 240 characters.
- Write files with LF line endings consistently; git here has core.autocrlf=true.
- W&B: project new_experiments, group stress-v3 (stress-v3-smoke for smoke), entity from the logged-in default. Laptop runs use offline/disabled W&B.

Reply with a short summary of what you understood and the files you will read first. Do not edit anything yet.
```

## Prompt 1: design the stress protocol (stop for approval)

```text
Design the stress-test protocol. Write it to run_scripts/stress/STRESS_PROTOCOL.md. Do not implement launchers or training code in this step; small read-only scripts to measure things are fine (put them under analysis/stress_design/ and use diagnostic seeds 401-499 / episode seeds 6,000,000+ only, never stress or final-comparison seeds).

The document must fix:

1. The temporal-delay axis for each environment, one primary axis per environment, with all other settings at the v3 main condition. Candidates from the proposal (section 4.3.3, quoted in BASELINE_PROTOCOL.md section 13):
   - Delivery: num_packages (the hidden delivered count must be remembered across more deliveries), grid_size, carrying capacity.
   - Colour: retention_delay_steps (the pure delay between the last cue and navigation), sequence length, repeated colours.
   - Maintenance: readiness_delay_range (stages between collecting a code and servicing it), active requests, distractors. readiness_delay_range must satisfy 1 <= min <= max <= max_stages - 1, so harder levels may need max_stages raised; say what else that changes.
   Prefer the axis that changes temporal delay most directly and changes least else. Justify each choice against RQ2.
2. The four levels per environment (easier, default, harder 1, harder 2) with exact values. The default must equal the v3 main condition exactly (same env kwargs and config hash as tuning's main condition apart from budget and seeds).
3. For every level: evidence it is solvable (scripted oracle or BFS success rate on at least 200 diagnostic seeds), the oracle's episode length distribution, and the episode step limit. The v2 rule was limit >= 2 x oracle p95 length. If a level needs a longer limit, decide whether to raise it for that level only and explain the confound.
4. Reward machine and agent validity at every level: the delivery RM scales with num_packages; QRM-Cue refuses fixed wall colours or a vocabulary below 4 (agents/reward_machines/colour.py) and its symmetric-room abstraction must still hold; DQN-TS task-state features and PPO-RM state sizes must match the level. Add or name the tests that prove RM reward equals env reward on every step at each level.
5. Training budgets: full v3 budgets (delivery 1,000,000; colour 3,000,000; maintenance 2,048,000) at every level. Evaluation as v3: 21 points (step 0 and every 5%), periodic episodes 50/50/20, final 200/200/100, greedy headline, noisy secondary with epsilon 0.05.
6. How tuned hyperparameters carry to the full budget. Tuning decayed epsilon over a fraction of the 25% budget, with warm-up at the absolute v3 step count. State the rule you will use (my default: the same selected fraction applied to the full budget, warm-up unchanged), check what resolve_config already does, and list it as a limitation.
7. Seeds and episode seed blocks: S2 above. Define sub-blocks for periodic vs final evaluation inside the 4,000,000 and 5,000,000 blocks the same way tuning_terminal sits inside 3,000,000, without renumbering anything existing.
8. The run matrix and counts: 11 cells x 5 seeds at default plus 11 cells x 3 seeds x 3 levels. Confirm the total (should be 154 runs).
9. Metrics and analysis plan, fixed now:
   - RQ1 (default level only): greedy success, progress and raw-return AUC (trapezoid over the 21 points, divided by budget), final values, steps to reach preset progress thresholds, IQM with stratified bootstrap 95% CIs (Agarwal et al. 2021). Main contrasts: explicit (QRM, QRM-Cue, PPO-RM) vs implicit (DRQN, Recurrent PPO), with memoryless (DQN, PPO) and task-state controls (DQN-TS) as reference points.
   - RQ2: performance at each level per cell, the degradation curve against the delay value, a normalised drop relative to the easier level, and slopes with CIs. Say how floor effects (everything at zero) will be reported so they are not read as "no degradation".
   - Noisy evaluation is reported next to greedy, never instead of it.
10. Machine plan: three machines, one process each, colour needs CUDA, others on CPU. Balance by expected hours, spreading the colour runs across all three machines. Use seed-major order so a partly finished machine still covers every cell. Give per-machine expected hours using measured tuning and baseline throughput, and say how --threads will be set (measure 1 vs 4 threads on the laptop for one delivery and one maintenance config if useful).
11. Disk and RAM per machine (colour replay about 2-4 GB per process), and what is kept: final policy per run plus CSV/JSON; restart data pruned after verification.
12. A deviations and limitations list carried from tuning (1 thread, mixed GPUs, NumPy workaround on one machine, 25% tuning fidelity, weak selection where finalists overlap) plus anything new.

End with a decision table of everything I must approve, with your recommendation for each. Then stop. Do not freeze or implement anything until I reply.
```

## Prompt 2: freeze the plan and implement configs

```text
I approve STRESS_PROTOCOL.md with the changes in my reply above (if none, as written). Record my approval as a dated decision (D11 or the next free number) in run_scripts/baseline/BASELINE_PROTOCOL.md section 11/13 and in STRESS_PROTOCOL.md. The easier level is part of that decision.

Implement, without starting any production run:
1. Level definitions as named environment variants or overrides in one place (extend agents/common/env_factory.py without changing the main condition or existing ablations). Default level kwargs must be byte-identical to the v3 main condition.
2. Seed handling: allow stress training seeds 201-299 and final-comparison seeds 301-399 only for phase "stress", with the episode seed sub-blocks from the protocol. Reject any other combination. Do not widen baseline or tuning seed permissions.
3. A stress plan generator (run_scripts/stress/generate_plan.py) that writes run_scripts/stress/stress_plan_v3.json: every run (env, algo, level, seed, machine role, slot order), the full resolved config, config hash and a plan hash. Hyperparameters come verbatim from run_scripts/tuning/results/v3/full/selected_configs.json, with the budget rule from the protocol. No random sampling. A --check mode rebuilds the plan and requires exact equality; on mismatch it prints the differing fields.
4. Config resolution for phase "stress" (resolve_config / runner flags) so a run can be started from a plan entry. Results go under run_scripts/stress/results/v3/<full|smoke>/<env>/<algo>/<level>/sNNN/aNN/, W&B group stress-v3. Keep paths under 240 characters on Windows from this checkout.
5. Tests (tests/test_stress_*.py): plan is deterministic and matches selected configs exactly; default level equals the main condition; each level's kwargs are valid and solvable on a few diagnostic seeds; RM reward equals env reward at each level; QRM-Cue and DQN-TS/PPO-RM state sizes match each level; seeds never leak between phases; path lengths; baseline and tuning regression fixtures unchanged.

Run the full test suite, python -B -m agents.common.baseline_plan write --check, the baseline manifest check (55/55), the tuning trial-list check, and the new plan --check. Report what changed, test results and the plan hash. Stop.
```

## Prompt 3: launchers, collection and smoke rehearsal

```text
Build the stress execution and collection pipeline under run_scripts/stress/, mirroring run_scripts/tuning (same patterns, separate files and results):
- run_a_stress.sh, run_b_stress.sh, run_c_stress.sh and a shared lib_launcher.sh: one process at a time, explicit cloud Python and explicit --threads, CUDA required for colour and refused otherwise, --continue-batch to skip verified runs after an interruption, BATCH COMPLETE / BATCH FAILED / PAUSED banners, nonzero exit on any failure, restart data pruned only after independent verification.
- preflight.py: interpreter, packages against requirements.txt, GPU and CUDA, free disk and RAM against the protocol's numbers, W&B login, plan --check on that machine's CPU (this is where the NumPy problem appeared during tuning), and a tiny smoke training for one cell of that machine.
- check_manifest.py with --role and --level filters, collect.py (CSV/JSON only, plus a content-addressed manifest), merge_bundles.ps1 (hash check, byte-exact conflict refusal), and a script that packs final policies into a tar.gz outside the checkout with a .sha256 file.
- A time estimate per machine printed at launch and updated after each run.

Then rehearse everything locally in smoke mode (labelled, results/v3/smoke, W&B offline): all three roles, an injected failure and a --continue-batch restart, collection, laptop merge in a different order than the roles ran, the manifest check, and model packing. Write rehearse.ps1 so I can rerun it. Do not use stress or final-comparison seed results for anything except pipeline checks; smoke runs are excluded from analysis.

Report the commands you ran and their output, and anything that only a real cloud machine can check. Stop.
```

## Prompt 4: production launch guide

```text
Write run_scripts/stress/PRODUCTION_LAUNCH.md for me to follow by hand. Use the style of run_scripts/tuning/PRODUCTION_LAUNCH.md as I edited it: plain English, separate exact steps for machine A, B and C, every step says what I should see, and finish lines written like "Finished when the terminal prints **`BATCH COMPLETE`**".

It must cover, per machine:
1. Rent: GPU model (RTX 3090 preferred), disk and RAM minimums, the image, and a table where I paste each machine's instance ID and SSH command. Leave clear <PLACEHOLDERS>.
2. Setup: clone new_experiments at the exact commit, create the cloud Python environment, wandb login, run preflight, then start the batch in tmux with the exact command. Include how to detach, reattach and check progress (status file and tail of the log) without stopping the run.
3. While running: what to check and how often, how long each machine should take, and what to do for BATCH FAILED, PAUSED, a dropped SSH connection, or a preflight plan mismatch (including the NPY_DISABLE_CPU_FEATURES fallback and the fact that it must be recorded).
4. When that machine prints BATCH COMPLETE: check and pack on the machine, download and verify the hash on my laptop, merge and verify that machine's results on the laptop, pack and download its final models to $HOME\Downloads\stress-models (outside the repository) with a hash check, then DESTROY that machine immediately. Each machine is finished and destroyed on its own, without waiting for the others. Nothing is destroyed before its downloads verify.
5. Laptop, once all three are in: final merge, full manifest check (154/154), then stage only CSV/JSON results and stop for me to review before commit. Never stage models, checkpoints, zips or .pyc files.
6. A final Vast.ai cleanup check.

Add rehearsal markers to the laptop PowerShell blocks and make rehearse.ps1 run them, so the guide's own commands are tested. Syntax-check every bash and PowerShell block. Report what you verified. Stop.
```

## Prompt 5: verify results and answer RQ1 and RQ2

```text
All stress runs are collected and merged on my laptop under run_scripts/stress/results/v3/full. Write the analysis under analysis/stress_v3/ following the analysis plan frozen in STRESS_PROTOCOL.md. Do not change the plan because of what the results show; anything extra is labelled exploratory.

1. Reconcile first: expected vs found runs, config and plan hashes, exact budgets and update counts, evaluation points and episode seeds, seed independence, resumes and failures, hardware and threads per machine. Missing or failed runs are reported as unavailable, never zero. Stop and tell me if anything does not reconcile.
2. RQ1 from the default level (seeds 301-305): learning curves, AUCs, final values and thresholds per cell with IQM and bootstrap CIs, and the explicit vs implicit contrasts with memoryless and task-state controls as references.
3. RQ2 from all four levels: degradation curves per cell and environment, normalised drops and slopes with CIs, and an honest treatment of floors (state plainly where every level is at zero).
4. Behaviour checks that explain the numbers (episode-end causes, wall touches, stage exits, progress) and a comparison with the tuning and v2 baselines, labelled as different protocols.
5. REPORT.md: results, limitations and deviations (including those carried from tuning), with figures. Then say whether the optional seed-expansion rule would trigger and roughly what it would cost.

Report the main findings in plain language first. Stop before committing.
```

## Prompt 6 (optional): seed expansion

```text
Only if I decide to spend more: apply the pre-registered seed-expansion rule from BASELINE_PROTOCOL.md section 6.3 as recorded in STRESS_PROTOCOL.md. Seeds come from the same range as the runs they extend (306+ for the default level, the next unused stress seeds for other levels), added to every algorithm of an environment together. Extend the plan with a new plan hash without changing existing entries, reuse the same launch guide with the new run list, and rerun the analysis with the extra seeds clearly reported. Stop before any rental.
```
