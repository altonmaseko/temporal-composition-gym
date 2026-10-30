# Tuning v3 selection (full)

Plan: `e160f2cf0fdacdca89218135287bafc9aefcbcd21d695582365da2114a857ddd`

Complete: True. Incomplete-data override: False; reason: None.
Selection uses confirmation seeds 102–104 only.

| Cell | Candidate | Greedy success / progress / return AUC | Seed SD | Noisy AUC (report only) | Tie-break |
|---|---|---|---|---|---|
| delivery/dqn | 00 | (0.0, 0.0365, -0.9996666666666667) | (0.0, 0.00916515138991168, 0.26128161690661167) | (0.0, 0.20416666666666666, -23.956333333333333) | greedy_progress_auc |
| delivery/dqn_ts | 07 | (0.0, 0.07683333333333334, -1.906) | (0.0, 0.011150485789118484, 0.4328882650292105) | (0.0, 0.3115, -26.41916666666667) | greedy_progress_auc |
| delivery/drqn | 07 | (0.0, 0.2803333333333333, -0.39666666666666667) | (0.0, 0.04284370821174719, 0.8637728771693016) | (0.006500000000000001, 0.6861666666666667, -31.737666666666666) | greedy_progress_auc |
| delivery/qrm | 07 | (0.0, 0.007, -1.9146666666666667) | (0.0, 0.006144102863722254, 0.6852091529842063) | (0.0, 0.10683333333333334, -26.812166666666666) | greedy_progress_auc |
| colour/dqn | 01 | (0.0, 0.014166666666666666, -0.7078333333333333) | (0.0, 0.0035472994422987934, 0.06779810715155206) | (0.0, 0.23316666666666666, -1.7226666666666666) | greedy_progress_auc |
| colour/dqn_ts | 04 | (0.0, 0.023666666666666666, -0.7248333333333333) | (0.0, 0.006211548384528074, 0.06566264793117407) | (0.0011666666666666668, 0.2748333333333334, -1.7208333333333332) | greedy_progress_auc |
| colour/drqn | 05 | (0.0, 0.0135, -0.769) | (0.0, 0.0005000000000000004, 0.05500681775925599) | (0.0003333333333333333, 0.22783333333333333, -1.7365) | greedy_progress_auc |
| colour/qrm_cue | 06 | (0.00016666666666666666, 0.030166666666666668, -0.4903333333333333) | (0.0002886751345948129, 0.010066445913694332, 0.022373719702663098) | (0.0018333333333333333, 0.3188333333333333, -1.5088333333333335) | greedy_progress_auc |
| maintenance/ppo | 01 | (0.0, 0.005, -3978.670958333368) | (0.0, 0.0025, 209.92079254326117) | (0.0016666666666666668, 0.028333333333333332, -1254.8878333333182) | greedy_progress_auc |
| maintenance/recurrent_ppo | 01 | (0.0, 0.0, -3842.4468749997886) | (0.0, 0.0, 272.6202092677585) | (0.0, 0.0, -3379.4522083331512) | greedy_return_auc |
| maintenance/ppo_rm | 04 | (0.0, 0.005833333333333334, -2498.4499583333395) | (0.0, 0.010103629710818452, 545.4000341538657) | (0.0016666666666666668, 0.01375, -650.2611666666631) | greedy_progress_auc |

Unavailable results are never zero. Per-seed variability, attempts and screening failures are recorded in selected_configs.json.

## Production review: deviations and limitations (2026-10-30)

Reviewed for production-launch step 7.4 against `TUNING_PROTOCOL.md`, the updated launch guide, and the collected production records. These notes supplement the generated selection table; they do not change the selected configurations or numeric results.

### Verified completeness and selection

- Independent verification passes for **154/154 runs**: 88 screening and 66 confirmation, with combined role counts A 27, B 74 and C 53. No unexpected runs or incomplete-data override are present.
- Recomputing screening rankings reproduces the frozen top-two finalists in all 11 cells. Recomputing confirmation means and sample standard deviations from the verified records reproduces all 11 selected candidates and their reported greedy AUC tuples. Selection uses confirmation seeds 102–104 only.
- Verification checks the registered configuration and plan hashes, exact training budgets and learner update counts, evaluation cadence and episode seeds, finite metrics, and final-evaluation records. Colour runs used CUDA; delivery and maintenance used CPU, as planned.

### Execution deviations and runtime differences

1. **Threads: 1 instead of the planned 4.** All 154 initial `metadata.json` files and the one `metadata_resume_*.json` file record `threads.requested = 1` and `threads.torch_num_threads = 1`. Thus all 155 recorded execution sessions used the same requested/PyTorch thread setting. This changes compute throughput relative to the plan; comparisons still use matched environment interactions. The metadata does not establish the thread count of every third-party library.
2. **Mixed GPUs instead of three RTX 3090 rentals.** Hardware recorded in the run metadata was:

   | Stage | Role A | Role B | Role C |
   |---|---|---|---|
   | Screening | RTX 4080 SUPER | RTX 4060 Ti | RTX 4070 Ti SUPER |
   | Confirmation | RTX 3090 | RTX 4070 Ti SUPER | RTX 4070 Ti SUPER |

   Confirmation used fresh rentals, as specified in the updated launch guide. Report hardware differences when discussing wall-clock time; these runs are not a hardware-controlled timing comparison. Configuration hashes and interaction budgets remain those of the frozen plan.
3. **Python runtime differs from the requirements file's reference environment.** All 155 execution-session metadata files record Python **3.12.15**, whereas the `requirements.txt` header describes a Python 3.10 reference environment. All ten dependency versions captured in the run metadata match their requirements pins: torch 2.14.0, stable_baselines3/sb3_contrib 2.9.0, gymnasium 1.3.0, minigrid 3.1.0, pymunk 7.3.0, pygame 2.6.1, numpy 2.2.6, wandb 0.30.0 and psutil 7.2.2. This is not a verification of every installed package. The recorded PyTorch CUDA version is 13.0 throughout.
4. **One interrupted screening run was resumed.** Machine B's `delivery/dqn/t06/s101/a00` has an earlier batch row with exit code 130. Its resumed session restored step **25,000**, then completed the registered **250,000** steps with **30,000** learner updates and passed independent verification. The pre-resume files and resume metadata/status are retained. This is one recovered run, not an additional independent trial; continuation is an episode-boundary restart, not a claim of bitwise uninterrupted training.

### Provenance and approved operational changes

Screening metadata records Git revision `9bfa221365b1e34ac66689b02dcd21052939ad6e`; confirmation records `b6f91363d79cc014ba63d72c8d8a7de77059cf06`. All 155 metadata records report a clean tracked worktree. Comparing both revisions to launch anchor `fe837314e40662649e0fe1bbfb7d88cfe2307d5b` finds no differences in the launch guide's checked Python/shell/PowerShell files, requirements/setup files, or frozen tuning/baseline plans. Different revision identifiers therefore do not indicate changes to those checked implementation files.

The spending cap became advisory by the dated protocol decision; the updated confirmation guide subsequently removed screening-hour carry-over and the combined spending cap. The updated guide defines confirmation's automatically saved start timestamp as its first setup-command time, rather than the earlier rental time. These are operational/accounting changes, not changes to training fidelity. No claim that the original combined 100-hour/$30 target was met can be made from this review. Verified-completion restart/replay pruning was also an approved storage change; it retained final policies and CSV/JSON evidence.

### Interpretation limits, not additional protocol deviations

- The registered tuning budgets were deliberately **25%** of the full v3 training budgets: delivery 250,000, colour 750,000 and maintenance 512,000 interactions per run. Exploration decay is scaled to that reduced budget while warm-up retains its absolute step count. These are planned choices; the search can miss late learners and does not establish full-budget superiority.
- **10 of 11 selected cells have zero mean greedy success AUC.** Colour QRM-Cue is the exception, with mean success AUC `0.00016666666666666666` (about **0.0167%**, after expressing the success-rate fraction as a percentage). All **33 selected-candidate confirmation runs** nevertheless have **zero final greedy success** in terminal validation. A nonzero periodic AUC must not be described as a nonzero terminal success rate.
- Ten cells were decided on greedy progress AUC; maintenance Recurrent PPO was decided on greedy raw-return AUC. These are the registered tie-breaks, not evidence of reliable task completion. Noisy evaluation remains secondary and was not used to choose the winners.
- `usable_for_final_comparison: true` means the full-mode tuning selection is available for subsequent experiments. The tuning observations themselves are not the tuned RQ1 comparison or stress-test results; those require the separately registered full-budget runs and held-out seed ranges.

This section is a manual review annotation. Re-running `select_configs.py` regenerates this Markdown report, so retain/reapply the review notes if selection is regenerated before publication.

