# Historical hyperparameter-tuning validity assessment

25 October 2026

## Verdict

The checkpoint-isolation fix is real, and this phase contains substantially more evidence of learning than the failed baseline repetitions. However, the QRM and DRQN **reported hyperparameter rankings are not trustworthy as currently assigned**, because their parallel objectives read the newest matching run folder rather than the folder belonging to the current trial. Independent numerical inconsistencies corroborate this source-level defect.

The final PPO-RM study is different. Its sequential in-process objective avoids that folder-selection error, and its 20 finite completed Optuna scores support a limited conclusion: trial 8 had the best recorded validation objective among those 20 trials for the implemented, 1,000-step, modified-reward task. The score is not a final-policy test estimate, a successful-maintenance rate, or a valid direct improvement estimate over the unbounded baseline.

| Study | What is supported | Principal restriction | Status of reported winner |
|---|---|---|---|
| QRM / Hazardous Delivery | 15 completed trials; optimization metrics in all 15; seed-specific checkpoint paths | Concurrent trials can receive another run's training-tail score | Trial 9 / -15.1 is stored correctly, but not validated as the best configuration |
| DRQN / Sequential Colour | 15 completed trials; optimization metrics in 12; three trials below replay warm-up | Same score-attribution defect; claimed winning score is inconsistent with its own run | Trial 12 / -5.8 is not a valid own-run last-ten objective |
| PPO-RM / Deferred Maintenance | 20 completed, sequential trials with a bounded callback evaluation objective | Best-of-checkpoints selection, no independent seed replication, modified reward and limited RM abstraction | Trial 8 / -434.2 is usable as the best observed selection score, with qualifications |

## Scope and provenance

- QRM and DRQN source snapshot: `2e6b0f3d26480339730e30355b051c72914a9a0d`, message `hyperparameter tuning fix`.
- Final PPO-RM source snapshot: `07c3656d9af2491116b6c7c945d7bc6610c74f04`, message `ppo tuning fix`.
- Both commits and their metadata were inspected locally. The supplied execution-to-commit mapping is used for this audit. Optuna start times are consistent with the stated UTC+02:00 times, but commit timestamps alone do not prove which checkout was deployed; neither the trial exports nor the inspected database rows store a per-trial Git SHA.
- All three SQLite databases were opened using `mode=ro` and `PRAGMA query_only=ON`. Trial states, values, decoded parameter distributions, and start/completion times were reconciled against the three result CSVs.
- Each of the 50 study trials was joined uniquely to a W&B record using algorithm, trial number, parameter equality and execution time. W&B timestamps have second precision, so a one-second tolerance was allowed at the trial start boundary. Ten additional preliminary/verification W&B records were excluded from study conclusions.
- Database start/completion intervals show peak overlap of eight QRM trials, two DRQN trials and one PPO-RM trial, matching the supplied historical concurrency settings.
- Naive database timestamps were treated as UTC, consistent with the corresponding W&B `created_at` values. For example, the first PPO trial starts at 19:14:18 UTC, corresponding to 21:14:18 in Johannesburg.
- The databases contain 15 QRM, 15 DRQN and 20 PPO-RM trials, all `COMPLETE`, all maximizing their returned objective, with no intermediate values or trial user attributes storing score-origin provenance. Database version metadata records Optuna 5.0.0; the story's generic reference to Optuna 3.x should not be copied as an exact software version without verification.
- Historical source was extracted into a separate audit folder. No original source, result CSV, database, GitHub branch or trained policy was edited. No training or tuning was run.
- The local full TensorBoard event logs, original cloud stdout logs and matching trial-policy checkpoints were unavailable. Large custom-agent CSV histories are sampled, which limits post-hoc recovery of exact own-run tail statistics.

Machine-readable evidence: [trial reconciliation and source hashes](tuning_export_audit.json), [historical logic probes](tuning_logic_probes.json).

## 1. Improvements that were actually implemented

### Seed-specific checkpoint selection

At `2e6b0f3`, QRM and DRQN use checkpoint names containing `seed{seed}` and pass the requested seed into the selector. Tuning launches each subprocess with `seed=trial.number`. This excludes the old unseeded baseline step-checkpoint names and checkpoints for another trial number.

A synthetic probe of the actual historical functions confirms:

- An old unseeded baseline checkpoint is ignored.
- A checkpoint for seed 1 is ignored when seed 2 is requested.
- An existing checkpoint for the same seed still resumes normally.

The matched official QRM and DRQN histories begin at low environment-step coordinates, rather than the late checkpoint offsets seen in the baseline audit, and contain optimization metrics for every QRM trial and 12 DRQN trials. This supports genuine new trial training rather than repetition of the baseline checkpoint-skip pattern. It is still important not to equate seed-specific naming with an unconditional guarantee of fresh training if the same trial numbers are reused with old files present.

### DRQN concurrency cap

The historical driver uses `n_jobs=2`, as supplied. That is an actual reduction in concurrent subprocesses. It does not address the separate question of which run supplies the score to each Optuna trial.

### Final PPO-RM fixes

The final factory wraps both training and evaluation environments with a 1,000-step `TimeLimit`. A non-training probe of that exact historical factory confirmed truncation on step 1,000 for a stationary episode. Thread clamping, CPU execution, sequential `n_jobs=1` and the native W&B callback are present.

The delivered final PPO database contains 20 completed trials, no stale `RUNNING` trial, and no fallback value of -9999. This is consistent with a successful final rerun. It does not independently prove the precise causes of the earlier freeze or the historical deletion procedure for the previous database.

## 2. QRM and DRQN return scores from a shared newest-folder selector

Both historical objective functions:

1. Launch training for the current `trial.number` and wait for that subprocess to exit.
2. Glob **all** matching algorithm/environment run directories.
3. Select `max(run_directories, key=os.path.getctime)`.
4. Read the selected directory's last ten completed training-episode returns.

They do not restrict the glob to the current trial number, capture the child process's run path or save a trial-to-score-source identifier in Optuna.

With eight QRM workers or two DRQN workers, a later-started trial can still be training when an older trial finishes. The older trial then reads the newer trial's partial log. Conversely, a late-finishing trial can read a different trial's already completed log. The database accepts the returned number as the current trial's value regardless of its source.

The audit executed the actual historical objective functions with simulated subprocesses and two competing run directories. In both algorithms, trial 0 launched correctly but returned trial 1's score. No training was executed in this probe.

This is separate from checkpoint isolation. Training weights can be independent while objective assignment is wrong. It also explains why an Optuna state of `COMPLETE` cannot certify that the value belongs to the associated parameters.

Source references: [QRM objective](snapshots/2e6b0f3/run_scripts/hyperparameter_tuning/tune_qrm.py), [DRQN objective](snapshots/2e6b0f3/run_scripts/hyperparameter_tuning/tune_drqn.py).

## 3. Numerical evidence that score assignment is wrong

### QRM: trial 1 cannot have its stated own-run last-ten score

QRM trial 1 stores **-45.75** in both the database and result CSV. Its matched W&B run is `rqfxu1qa`, and its sampled history alone contains 816 individual episode-return records.

The historical Hazardous Delivery defaults produce integral rewards and integral episode returns: delivery +10, completion +50, damage -1, and failure -50. An average of the last ten integral episode returns must be a multiple of 0.1. **-45.75 cannot be such an average.** There are clearly more than ten own-run episode records, so this is not explained by an own-run window containing only four episodes.

This inconsistency does not depend on recovering the last ten events from a sampled CSV. It establishes that at least one objective did not use the specified complete own-run terminal window. Reading another, newer trial's short partial window is consistent with the source-level selection defect.

The stored winner is trial 9, value **-15.1**, with:

- learning rate `2.206773917364734e-05`
- replay capacity `100000` transitions
- target-network frequency `2000`
- batch size `32`
- exploration fraction `0.40928673932827986`

These parameter values and the stored ranking reconcile across artifacts. The audit does not establish that -15.1 is trial 9's own last-ten mean or that trial 9 was the best-performing candidate. The affected objective process prevents accepting that interpretation without recovering the correct per-run scores.

### DRQN: the claimed winner is inconsistent with its own final return

The database and CSV select trial 12 at **-5.8**, uniquely matched to W&B run `of7weqyx` by time and all five parameters. That run's final recorded individual episode return is **-466**.

Under the executed sparse-reward environment, an episode can earn at most +10. Therefore, even if the preceding nine episodes were all perfect, the run's own last-ten mean would satisfy:

`mean <= (-466 + 9 x 10) / 10 = -37.6`

The reported **-5.8 exceeds this upper bound** and cannot be that run's own specified terminal last-ten objective. With fewer than ten episodes the upper bound would be even lower, though this trial has ample own-run returns. The bound uses the final summary scalar and the environment reward ceiling; it does not equate a sampled CSV's tail with the full TensorBoard tail.

Trial 2 also violates the corresponding bound: final return -429 gives an upper bound of -33.9, whereas its stored value is -10.3.

The reported trial-12 parameters are:

- learning rate `1.2570041324220499e-05`
- replay capacity `10000` episodes
- sequence length `4`
- batch size `16`
- exploration fraction `0.6433114610115815`

These are accurately recorded candidate settings, but **their reported winning score is not a valid own-run terminal score**. It is not defensible to conclude that a sequence length of four or this learning rate was optimal from that ranking.

## 4. DRQN did learn in most tuning trials, but three trials did not

This is an important distinction from the corrected baseline visual DRQN runs. Twelve of the 15 tuning runs contain TD-loss/Q-value records, showing that optimization was reached.

Three have complete small histories, no optimization metrics, and too few completed replay episodes:

| Trial | Run ID | Batch size | Episodes needed to start updates | Completed episodes logged | Stored objective |
|---:|---|---:|---:|---:|---:|
| 3 | `tgqnvkri` | 32 | 33 | 24 | -164.1 |
| 5 | `90jk6myc` | 64 | 65 | 44 | -122.9 |
| 9 | `0hl7tp63` | 64 | 65 | 26 | -141.5 |

For these three complete histories, the exact own-log last-ten means do match their stored objectives. Their problem is failure to reach learning warm-up, rather than an observed mismatch in those particular scores.

For the other trials, a small number of episode rows in a sampled CSV must not be used to infer that learning failed: the recorded loss metrics independently show optimization occurred. The audit makes this distinction explicitly.

The source still has unbounded visual episodes. An unfinished episode may consume the rest of a 300,000-step budget, while the objective averages older completed episodes. Thus even a corrected trial-specific path would leave a noisy, potentially stale training-window criterion rather than a clean evaluation of the final learned policy.

## 5. PPO-RM: usable limited selection result, not a confirmed final-policy score

### What is supported

The sequential objective directly owns its model and `EvalCallback`; it does not read the newest global run directory and does not load a baseline checkpoint. Each trial constructs a new PPO model.

All 20 values, parameter sets and timestamps agree between the read-only database and CSV. Their matched W&B records are marked `finished`. Trial 8 is correctly the largest stored value, **-434.2000000000001**, conventionally reported as **-434.2**. Its configuration is:

- learning rate `0.0008086919694109821`
- rollout length `4096`
- batch size `128`
- entropy coefficient `0.02573863841679592`
- clipping range `0.2`

The executed objective trains for nominally 500,000 steps, evaluates every 50,000 steps using deterministic actions, and returns `eval_callback.best_mean_reward`. The default callback uses five evaluation episodes per evaluation. Consequently, the selection value is the **best five-episode evaluation mean observed across approximately ten checkpoints**, then maximized again across 20 hyperparameter trials. Normal on-policy rollout rounding can slightly exceed the nominal training budget.

The appropriate conclusion is that this configuration obtained the highest recorded **selection score in this search**. It is not an independently replicated estimate of expected performance or proof of a global optimum.

### Qualifications required in the report

1. **Best checkpoint is not final checkpoint.** The returned maximum need not describe the final model after training. The script does not configure a best-model save path or save a final trial model, so it does not itself preserve the policy that achieved the selected score.
2. **Selection optimism remains.** Choosing the best noisy evaluation checkpoint and then the best trial can overestimate performance. No independent final test set or repeated training seeds validate -434.2 here.
3. **No explicit model/evaluation seed is supplied.** Deterministic actions do not make the generated stages, initialization or whole experiment deterministic. There is one stochastic training realization per parameter configuration, rather than a replicated average across seeds.
4. **The horizon differs from baseline.** Both training and evaluation now truncate at 1,000 steps; the baseline maintenance episodes had no time limit. A less negative reward can result from changing the horizon as well as policy quality. Directly comparing -434.2 to an unbounded baseline completed-episode mean is invalid.
5. **The objective is modified RM reward.** The wrapper derives rewards from simulator score increments and the outer Monitor records those modified rewards. A historical-factory diagnostic again produced raw +1100 but modified/monitored +1200 on a compound service-and-exit transition. Its frequency in actual trials is unknown; the observed objective cannot be converted to raw return by a universal correction.
6. **The RM represents readiness only.** Readiness already appears in the base observation; held maintenance codes are not encoded. This search tunes the implemented readiness-wrapper PPO, rather than validating complete explicit code memory.
7. **Negative return is not a success-rate measurement.** The stored objective does not distinguish complete maintenance from truncated low-penalty behaviour. Task success requires its own criterion.

The value **-1897.34** appears as PPO-RM tuning **trial 2's objective**. It should not be presented as an independently measured baseline from which the tuning improved to -434.2. The earlier baseline used different horizons, metrics and budgets.

Official callback semantics: [SB3 2.9.0 EvalCallback implementation](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/callbacks.html).

### Why the empty PPO-RM W&B histories mean something different here

All 20 final-study PPO-RM history CSVs are empty. This is a telemetry limitation, not the checkpoint-skip pattern found in the baseline.

The tuning script removes TensorBoard synchronization, sets no `tensorboard_log`, enables no gradient logging, and saves no callback model. The native W&B callback primarily supplies configuration/model/gradient integration; it does not by itself upload the SB3 scalar learning curves in this setup. The 20 completed finite callback objectives remain in Optuna, but their evaluation time series and episode-level distributions were not stored in the inspected database or exported W&B histories.

Some W&B configuration fields such as `_n_updates` or `num_timesteps` are snapshots of model attributes captured at callback initialization; a zero there must not be treated as proof that the whole trial did no training. This callback logs configuration before training proceeds.

The missing telemetry prevents independently reconstructing which checkpoint produced -434.2 or its episode-return variance from these files. Original per-run stdout logs could provide additional evidence if retained on W&B or the server.

Primary integration source: [W&B's official SB3 callback](https://github.com/wandb/wandb/blob/main/wandb/integration/sb3/sb3.py).

## 6. What can be recovered without rerunning training

### QRM and DRQN

Recover the complete per-trial TensorBoard scalar histories, or unsampled W&B histories for each correctly matched training run. The per-trial W&B IDs are recorded in `tuning_export_audit.json`.

Then:

1. Read each run's own final ten `charts/episodic_return` events.
2. Confirm optimization occurred, and separately identify the three DRQN trials that never reached warm-up.
3. Recompute a transparently corrected ranking of the candidates that were actually tested.
4. Preserve the original Optuna values and label the recomputation as a post-experiment correction.

The original TensorBoard directory ctimes and parent-process "Extracting reward from" output would additionally establish the exact source of each corrupted original score, but those are not necessary to compute a new own-run ranking from complete individual training logs.

Every trained custom-agent official history inspected here is a 1,000-row sampled export with gaps in W&B `_step`. Therefore, taking the last ten episode rows of those CSVs would **not** reliably recover the original last-ten scalar statistic. The complete small histories belong to the three non-learning DRQN trials; extra preliminary runs must not be substituted for official study trials with the same seed number.

Recomputing the scores could recover the best observed candidate without retraining, if complete histories are available. It would not make the original adaptive Optuna sequence correct retroactively: later parameter suggestions may have been influenced by wrongly assigned earlier values. Nor would it turn a single training-tail score into independently validated final-policy performance.

### PPO-RM

The database already supports its recorded limited ranking. Retained stdout logs could verify the sequence of callback evaluations and when the best score occurred. New independent evaluation would be useful if the exact candidate policies were saved elsewhere, but this script does not itself persist them. If no suitable weights survive, independent policy confirmation requires training again.

## 7. Deadline-safe reporting

For QRM and DRQN, describe the selected parameter sets as **candidate configurations from an exploratory search whose score attribution was later found to be unreliable**. Do not call them confirmed optimal or claim that they resolved instability, improved memory capacity or established an advantage for particular BPTT lengths. If full logs can be recovered, replace that qualification with a clearly documented corrected own-run ranking.

For PPO-RM, report trial 8 as the **best observed validation configuration in a 20-trial, 1,000-step readiness-wrapper search**, with -434.2 explicitly labelled as the maximum checkpoint-level five-episode mean. State the lack of repeated seeds and independent final testing. Keep its modified reward and changed horizon visible when comparing phases.

Suggested report paragraph:

> Hyperparameter searches comprised 15 QRM trials, 15 DRQN trials and 20 readiness-wrapper PPO trials. Seed-specific checkpoint selection prevented the baseline cross-seed loading pattern in the custom-agent searches. Post-experiment validation nevertheless found that the parallel QRM and DRQN objective functions selected the newest shared run directory rather than binding scores to the current trial, compromising their recorded rankings. These settings are therefore treated as exploratory candidate configurations pending reconstruction from complete per-run logs. The sequential PPO search used bounded 1,000-step episodes and selected the largest five-episode evaluation mean observed during training. Its best recorded value was -434.2 in trial 8; this is a search-selection statistic for the implemented modified-reward task, rather than an independently replicated final-policy performance estimate.

Suggested overall conclusion:

> The tuning phase established that many candidate agents performed optimization and that bounded evaluation resolved the nonterminating maintenance-evaluation problem. The available artifacts support a limited PPO search-selection result. They do not currently establish the reported QRM/DRQN optima, architecture-level memory conclusions or comparable improvement estimates over the original baseline phase.
