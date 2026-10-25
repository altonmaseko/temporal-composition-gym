# Historical baseline validity assessment

25 October 2026

## Verdict

The evidence does **not** support treating the exports as 24 independently trained, three-seed baseline experiments. Several individual traces remain useful, but the baseline comparison needs substantial qualification before it is used in a final report.

This conclusion comes from the two historical commits supplied by the user and the baseline exports, not from later hyperparameter-tuning or stress-testing code. It supersedes any baseline attribution inferred from the earlier audit of the current checkout.

The main problems are:

1. Seed-independent checkpoint names caused later seed-labelled jobs to load earlier weights. Eight SB3 jobs contain no training history, while four Hazardous Delivery DRQN/QRM jobs resume near 900,000 steps.
2. The six corrected jobs requested only 100,000 steps each, rather than the original 2-3 million-step budgets for their environments.
3. The three visual DRQN histories never accumulate the 33 completed replay episodes required to start optimization and contain no optimization metrics.
4. Several maintenance jobs trained but produced no completed-episode return observations. Other maintenance means are extremely sparse or stale, rather than reliable end-of-training performance estimates.
5. The corrected maintenance PPO-RM is a readiness wrapper, and its outer SB3 monitor records the modified reward. It does not provide the proposed explicit maintenance-code memory or guarantee the same return objective as PPO.

## Scope, provenance and preservation

- Initial 18 records: `596256061ffae5ea61dee31f2f5e70efc3bf67c7`, message `minor`, 30 September 2026, 13:53:04 +0200.
- Six corrected records: `1760a6c5c2a86997c864e39810c79ca4db79f005`, message `rerun fix`, 3 October 2026, 19:41:41 +0200.
- The mapping from runs to these commits is the user's supplied verification from historical W&B metadata. Both Git objects, their metadata, source trees and differences were independently inspected locally. The supplied CSVs do not themselves contain commit hashes or run exit status, and this audit did not independently query the live W&B API.
- Read the 24-row baseline summary and all 24 matching baseline history files, the baseline execution story, and the baseline launch scripts at the relevant commits. Hyperparameter-tuning and stress-testing results were not analyzed.
- Read-only Git operations extracted snapshots under this audit folder. No branch checkout, reset, push, original result edit, or training was performed. The existing modification to the hyperparameter-tuning story was left untouched.
- Historical Python modules were exercised in separate processes using `D:\miniconda3\envs\temporal_gym\python.exe`. The probes assert that the imported environment package comes from the selected historical snapshot. These checks reproduce code behaviour with the installed local libraries, rather than reconstructing the complete cloud software image.
- Local `models`, `runs` and `wandb` folders were empty, so no original cloud model weights, full TensorBoard event files or execution logs were available to verify saved weight identity or exit codes.

Evidence files:

- [Per-run export audit and original CSV hashes](baseline_export_audit.json).
- [Initial-commit diagnostic results](probe_5962560.json).
- [Fix-commit diagnostic results](probe_1760a6c.json).
- [Initial source snapshot](snapshots/5962560/agents/train_dqn.py).
- [Corrected source snapshot](snapshots/1760a6c/agents/ppo_rm/train_ppo_rm.py).

Original data: [summary CSV](../../../wandb_phase_3_baseline_exports/all_runs_summary.csv), [baseline execution story](../../../Phase%20Stories/%5B1%5D%20Phase%203%20Baseline%20Execution%20Story.md).

## What changed between the two baseline commits

The complete environment tree is unchanged between `5962560` and `1760a6c`. Therefore, the bug-fix commit does not by itself create a different task distribution for the six reruns.

Relevant source changes were:

- DRQN's forward pass gained explicit batch/time dimensions and image flattening.
- PPO-RM gained a branch for Deferred Maintenance, a maintenance proposition wrapper and a two-state readiness machine.
- A parallel rerun shell script was added.

The fixes address real execution defects. A local initial-commit image inference fails with `mat1 and mat2 shapes cannot be multiplied (7056x3 and 21168x64)`, while the corrected forward pass produces shape `(1, 1, 4)`. The initial PPO-RM construction for maintenance fails because the delivery reward machine requires `num_packages`; the corrected maintenance construction works.

Consequently, the earlier descriptions of image replay-memory allocation and continuous-action dimension incompatibility should not be treated as verified root causes without the original error logs. The source directly demonstrates an image-input shape defect and a delivery-specific RM construction defect. The image fix did not change replay storage, capacity or warm-up requirements.

## 1. Checkpoint collision destroys the intended independence of seed replicates

All baseline checkpoint selectors match algorithm and environment, but omit the seed. The seed-specific final model filenames do not fix the problem because the next job resumes from the separate shared step-checkpoint files.

### Eight SB3 records contain no new training history

The following jobs have zero history rows, no environment-step record, no return metric and a reported runtime of two seconds:

| Environment | Algorithm | Seed labels | Run IDs |
|---|---|---|---|
| Hazardous Delivery | DQN | 2, 3 | `px1khwjo`, `cdov4ojs` |
| Sequential Colour | DQN | 2, 3 | `91y6loyj`, `i21h8it2` |
| Deferred Maintenance | PPO | 2, 3 | `pdquwaab`, `vdf386xv` |
| Deferred Maintenance | Recurrent PPO | 2, 3 | `ncjwx8jg`, `vepj877j` |

The historical source explains this pattern. Seed 1 saves a shared checkpoint at the requested budget. Seed 2 loads it and computes `remaining_timesteps = requested - checkpoint_step`. That difference is zero, so `model.learn()` is skipped. Seed 3 follows the same route. Loading without an overriding seed also retains the model's stored seed; changing the run label and final filename does not create independent initialization.

The synthetic probes execute the actual historical selector functions with simulated checkpoint listings. They select the shared full-budget checkpoint and return zero remaining steps. Combined with the empty histories and runtimes, this is strong evidence of checkpoint-skip jobs, not three independent completed training runs. Actual file identity on the cloud disk cannot be verified from the available exports alone.

### Four custom-agent records are continuations, not fresh seed replicates

DRQN and QRM save at step 900,000 before finishing their million-step loops. Their loops end before a checkpoint at step 1,000,000 is written. Later seed-labelled jobs find the same 900,000-step checkpoint and train approximately another 100,000 steps.

| Algorithm on Hazardous Delivery | Seed label | First exported environment step | Last exported environment step | Runtime |
|---|---:|---:|---:|---:|
| DRQN | 1 | 1,000 | 998,255 | 8,696 s |
| DRQN | 2 | 900,115 | 999,910 | 765 s |
| DRQN | 3 | 900,203 | 999,900 | 792 s |
| QRM | 1 | 1,000 | 993,500 | 32,980 s |
| QRM | 2 | 900,499 | 999,953 | 3,360 s |
| QRM | 3 | 900,274 | 999,968 | 3,479 s |

Some histories are sampled, so these are first and last exported logging coordinates, not exact first and last executed actions. The step offsets and roughly tenfold runtime differences corroborate the source's shared-checkpoint behaviour.

The resumed jobs also rebuild empty replay buffers, reset environment and recurrent state, and refresh target networks from the loaded online network. They are not exact uninterrupted replays of seed 1. They may be studied as continuation experiments, but cannot be used as independent from-scratch seeds or concatenated into an original learning curve without disclosing the restart.

**Report consequence:** do not calculate an across-three-seed mean, standard deviation, confidence interval or significance test from these seed labels. For the original series, the available evidence supports individual seed-1 reference trajectories and separately identified continuations.

Historical references: [SB3 selector and skip branch](snapshots/5962560/agents/train_dqn.py), [DRQN selector and loop](snapshots/5962560/agents/drqn/train_drqn.py), [QRM selector and loop](snapshots/5962560/agents/qrm/train_qrm.py).

## 2. Corrected runs used different training budgets

The original launch scripts specify:

| Environment | Original intended budget per algorithm/seed |
|---|---:|
| Hazardous Delivery | 1,000,000 steps |
| Sequential Colour | 3,000,000 steps |
| Deferred Maintenance | 2,000,000 steps |

The corrected parallel launch script omits `--total-timesteps` for DRQN and `--timesteps` for PPO-RM. Both parsers default to 100,000. The exported configurations confirm 100,000 for all six reruns. PPO-RM records extend to 100,352 steps, consistent with completing a 2,048-step on-policy rollout beyond the requested lower bound.

This is a 30-fold budget difference between visual DQN and rerun DRQN, and a 20-fold difference between maintenance PPO/Recurrent PPO and rerun PPO-RM. Nominal job completion does not make these equal-budget comparisons.

An analysis restricted to an overlapping step interval can sometimes compare learning traces, but does not fix nonindependent seeds, missing returns, untrained visual DRQN or mismatched reward definitions. Do not simply scale returns by the budget or stretch a 100,000-step curve to the longer horizon.

References: [original memoryless script](snapshots/5962560/run_scripts/run_memoryless_baselines.sh), [original implicit script](snapshots/5962560/run_scripts/run_implicit_memory_baselines.sh), [original explicit script](snapshots/5962560/run_scripts/run_explicit_memory_baselines.sh), [corrected rerun script](snapshots/1760a6c/run_scripts/rerun_drqn_image_and_ppo_rm_continuous.sh).

## 3. Visual DRQN does not qualify as a trained recurrent baseline on this evidence

The replay buffer accepts an episode into its sampling pool only after that episode ends. Optimization requires `len(rb.buffer) > args.batch_size`. The batch size in these three jobs is 32, so at least 33 completed episodes are needed before the first parameter update.

| Rerun seed | Run ID | Completed episodes logged | Last completed episode ends at zero-based step | Loss/Q-value training metrics |
|---:|---|---:|---:|---|
| 1 | `ensf2p8c` | 23 | 22,062 | None |
| 2 | `nxjraogz` | 28 | 28,136 | None |
| 3 | `9dpye8xe` | 29 | 29,214 | None |

For each small history, W&B `_step` is contiguous from zero through its summary endpoint. Recorded episode lengths sum exactly to the last completed episode endpoint plus one. These histories are not the 1,000-row sampled exports found in larger jobs. They contain only episode-return and episode-length logging, with no `losses/td_loss` or `losses/q_values`.

The implemented replay gate stays closed for 23, 28 or 29 completed episodes, as confirmed using the actual historical replay-buffer class. On the available complete small histories, learning therefore never starts. The last logged episode ending near 22,000-29,000 is not proof that execution stopped there: the remaining configured steps can belong to an unfinished episode, with no new episode log. The absence of original final model weights and execution logs prevents independently verifying the final loop counter or saved initial-weight equality.

This evidence does not support calling the six rerun jobs failed: corrected visual inference works and a process may complete its configured loop. It does mean that these three visual records should not be accepted as evidence about a trained LSTM's performance without contrary optimizer or weight evidence.

The visual DRQN also flattens raw RGB values into a linear layer, whereas visual DQN uses SB3's CNN policy and image preprocessing. Even a properly trained rerun would compare more than the presence of memory alone.

**Report consequence:** label these three records as visual DRQN attempts that did not reach the learning warm-up threshold, rather than trained DRQN baselines. Their returns cannot show recurrent memory degradation or catastrophic forgetting. Obtaining a trained visual DRQN comparison requires actual parameter updates; relabelling a plot cannot provide them.

## 4. Missing and stale maintenance episode returns

At both baseline commits, Deferred Maintenance has no registered episode time limit and returns `truncated=False`. The agent scripts add no `TimeLimit`. Reaching the total training budget does not force the current episode to terminate and be logged.

The exports show:

- PPO seed 1 records optimization through 2,000,896 steps. Its episode-return mean is exactly -5173.8 and episode-length mean exactly 2314 at all 976 nonblank return records. This is a repeated completed-episode buffer statistic, not 976 independent episodes or held-out policy evaluations.
- Recurrent PPO seed 1 records optimization through 2,000,896 steps but no episode-return or episode-length metric. Its performance is unavailable from these exports; it is not zero.
- Corrected PPO-RM seed 1 records optimization through 100,352 steps. It has 48 logged moving means but only three distinct mean values. Its final mean is -5219.1333.
- Corrected PPO-RM seeds 2 and 3 record optimization through 100,352 steps but no episode-return metric. Their training records are nonempty; their episodic-return performance is unavailable.

For the jobs with no episode metric, the source and logs are consistent with an episode that never completed within the training budget. For a constant mean, it is not possible to recover the exact number of underlying completed episodes from the aggregate alone. A flat curve must not be interpreted as stable asymptotic performance without further evidence. Several maintenance jobs may simply continue within a long episode while the logger repeats old completed-episode information or has none to report.

**Report consequence:** omit an unsupported PPO/Recurrent PPO/PPO-RM ranking from these records. Leave missing returns as unavailable, never zero. A separate finite-horizon evaluation of the original frozen policies could supply a new evaluation result if those policies can be retrieved, but would not reconstruct the missing baseline learning curves or create independent training replicates.

## 5. What the baseline PPO-RM actually represents and logs

The corrected maintenance RM has two states: waiting and ready. The 25-element base observation already includes readiness. The appended two-element one-hot does not encode the held maintenance codes or supply the missing history that the proposed explicit method was intended to track. Thus the implementation is appropriately described as PPO with a two-state readiness wrapper.

Its reward payload is the change in the simulator's score, and `PPORMObservationWrapper.step()` replaces the raw step reward with that RM reward. Score increments can differ from raw reward when multiple reward-bearing events coincide.

An important correction to the earlier general audit: an inner `RecordEpisodeStatistics` wrapper does **not** guarantee that PPO-RM's SB3 return is the raw environment return. SB3 automatically adds an outer `Monitor`, which sums the rewards it receives and overwrites `info['episode']` on termination. Here those rewards have already been replaced by PPO-RM.

The historical fix-commit probe constructs the actual wrapper stack and a synthetic simultaneous service/final-exit event:

| Quantity | Diagnostic value |
|---|---:|
| Inner raw-environment episode return | +1100 |
| PPO-RM modified reward | +1200 |
| Outer Monitor's recorded episode return | +1200 |

This proves that the metrics can use different reward definitions. It does not establish how often such compound transitions occurred in the actual baseline jobs, and it is not a correction factor that can be subtracted from the exported means. PPO-RM's three sparse mean plateaus cannot be converted into raw PPO returns from these exports alone.

**Report consequence:** describe the readiness abstraction and modified reward explicitly. Do not claim that a performance difference demonstrates the benefit of explicit maintenance-code memory, or assume its rollout reward is automatically comparable to PPO's.

Official implementation references: [SB3 2.9.0 automatic wrapping](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/base_class.html), [SB3 2.9.0 Monitor reward recording](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/monitor.html).

## 6. Individual observations that remain supportable

### Hazardous Delivery

The seed-1 DQN, DRQN and QRM files provide individual training trajectories. They may support descriptive discussion of these specific runs, subject to different logging definitions and sampled histories.

- DQN seed 1 has a final reported moving return mean of -23.3. This does not prove that no individual episode ever succeeded.
- DRQN seed 1 has a final reported individual episode return of +29. The exported history's sampled maximum is +20, illustrating why a sampled history and final W&B summary need not coincide. Neither statistic is an estimate of stable convergence by itself.
- QRM seed 1 includes return +80, the maximum undamaged three-delivery plus goal return under the default rewards. Its exported raw-return samples include 12 values above +30. An episode without successful goal completion cannot exceed the three delivery rewards totalling +30, so these samples demonstrate actual full-task successes. They do not establish a population success rate, three-seed reliability or sustained optimality.
- The seed-3 DRQN continuation includes +76 and two sampled returns above +30. It also achieved complete episodes, but after loading prior training, rather than as an independent seed-3 experiment.

Default health and delivered count were visible, `task_dependencies=False`, and observations had nine values. The baselines therefore did not test the proposed hidden-delivery-count ablation or enforce ordered package dependencies. Local mapping and carried inventory remain limited, so this is not a claim of full observability; it is a restriction on which memory mechanism the comparison isolates. The delivery RM also tracks only the delivery count rather than all hidden task state.

The equal-penalty lava event ambiguity remains in this historical wrapper. Its two failure labels produce the same reward and terminal transition in the delivery RM; it does not invalidate these raw return observations by itself.

### Sequential Colour

DQN seed 1 provides a trained-policy training trace. The other two DQN labels have empty histories. The three corrected DRQN records fail the replay warm-up check described above.

With `hard_death_mode=False`, `infinite_mode=False`, the default sparse reward and no time limit, every logged completed episode in these drivers ends by finishing the sequence. Wrong walls incur penalties and allow another attempt. A negative completed-episode return therefore means eventual completion with many mistakes, not necessarily failure to finish.

The final DQN moving mean of -121.02 corresponds to approximately 131.02 wrong-wall contacts per completed sequence in that moving window because episode reward equals `10 - number_of_wrong_contacts`. This is poor efficiency, but it is not a zero-success measurement. Long unfinished episodes are absent from the completed-episode mean, creating censoring that must be acknowledged.

Visual DRQN's occasional +10 cannot show learned recall when its parameters have not reached the optimizer update threshold. This environment permits eventual progress through retries, so an untrained policy can complete episodes by chance. Do not use these results to prove that all memoryless policies are structurally unable to complete the task.

## 7. Earlier concerns that do not apply to these baseline runs

- No visual QRM baseline was run. The later visual RM event-aliasing problem is not a baseline finding.
- Baseline maintenance used `readiness_delay_range=(1,3)`, meaning a target of one or two completed stages, with four total stages. Readiness is reachable under both baseline targets. The later beyond-horizon delay-sweep problem does not invalidate the baseline configuration. This does not certify every generated layout's solvability.
- The zero-retention bug is inactive at the baseline default retention delay of two steps.
- The later 1,000-step time-limit wrapper is absent from both baseline commits.
- Inactive distractor/duration switches do not establish a failed baseline treatment because no such baseline intervention is specified by the historical launch commands.
- Module-level NumPy environment randomness and the custom agents' unseeded action spaces limit exact replay. They are secondary to the directly corroborated checkpoint collision when assessing the claimed seed independence.

## 8. Complete baseline disposition

| Environment | Algorithm | Seed 1 | Seed 2 | Seed 3 | What can be reported |
|---|---|---|---|---|---|
| Hazardous Delivery | DQN | Individual training trace | Empty history, checkpoint-skip pattern | Empty history, checkpoint-skip pattern | Single reference trajectory |
| Hazardous Delivery | DRQN | Individual training trace | 900k continuation | 900k continuation | Reference plus disclosed continuations |
| Hazardous Delivery | QRM | Individual training trace; observed full successes | 900k continuation | 900k continuation | Reference plus disclosed continuations |
| Sequential Colour | DQN | Individual training trace | Empty history, checkpoint-skip pattern | Empty history, checkpoint-skip pattern | Single reference trajectory |
| Sequential Colour | DRQN | 23 episodes, no optimizer metric | 28 episodes, no optimizer metric | 29 episodes, no optimizer metric | Attempts below learning warm-up, not trained DRQN comparison |
| Deferred Maintenance | PPO | Optimization, stale return mean | Empty history, checkpoint-skip pattern | Empty history, checkpoint-skip pattern | Limited descriptive training observation |
| Deferred Maintenance | Recurrent PPO | Optimization, no episode-return metric | Empty history, checkpoint-skip pattern | Empty history, checkpoint-skip pattern | Optimization occurred; return unavailable |
| Deferred Maintenance | PPO-RM | 100k budget, sparse modified-return mean | 100k budget, return unavailable | 100k budget, return unavailable | Limited readiness-wrapper prototype records |

Totals: 24 run records, 16 nonempty histories, 8 empty histories, 13 records with any episode-return metric, and 4 Hazardous Delivery continuations. These categories overlap; they are not counts of independent successful training experiments.

## 9. Reporting this work before the deadline

### Retain

- The benchmark engineering and the historical record of attempted baseline execution and remediation.
- The individual usable trajectories, each clearly labelled by run ID, seed label, commit, budget, metric definition and checkpoint origin.
- The observation that full Hazardous Delivery completions occurred in specific QRM and continued DRQN episodes.
- The finding that the initial infrastructure and logging design required correction before a controlled experiment could be claimed.

### Remove or qualify

- "24 successful independent baseline runs" and "three independently trained seeds per configuration."
- Three-seed error bars or significance claims derived from these labels.
- A trained visual DRQN versus DQN performance or memory-retention conclusion.
- A maintenance ranking based on missing values, stale completed-episode means and differently defined rewards.
- A claim that reactive policies cannot complete Sequential Colour: retry-based completions and completed-episode censoring contradict that interpretation.
- A claim that one peak return establishes convergence, or that the observed fluctuations specifically diagnose catastrophic forgetting.

### Recommended limitations paragraph

> Historical reconstruction of the baseline implementation identified execution and measurement limitations. Checkpoint discovery did not distinguish random seeds, causing some subsequent seed-labelled jobs to skip training or resume previously trained parameters. These records were consequently not treated as independent training replications. Corrected visual DRQN and maintenance PPO-RM jobs used shorter budgets than the original baselines; the visual DRQN runs did not reach the completed-episode replay threshold needed to begin optimization. Maintenance episodes lacked a finite truncation horizon, producing missing or stale completed-episode return statistics. The maintenance reward-machine implementation additionally represented readiness rather than held-code memory and used a modified reward. The available baseline results are therefore reported as exploratory implementation observations rather than a controlled three-seed comparison of memory architectures.

### Recommended baseline conclusion

> The baseline phase demonstrated successful interaction and, for selected runs, optimization and complete task episodes. It also exposed limitations in checkpoint isolation, learning warm-up and episode measurement. The retained trajectories provide descriptive evidence for the tested implementations, but do not establish a reliable architecture ranking or multi-seed sample-efficiency comparison. Independent initialization, verified parameter updates and a consistent evaluation protocol are required before stronger comparative conclusions can be drawn.

Fresh independent training is necessary if the original three-seed learning-curve claim must be retained. Re-analysis cannot manufacture missing independent replications or turn untrained policies into trained ones. If the deadline precludes that, the appropriate workaround is reduced claims and transparent reporting. An evaluation of preserved frozen policies could add behavioural evidence, but would not restore the missing baseline training experiment or sample-efficiency history.
