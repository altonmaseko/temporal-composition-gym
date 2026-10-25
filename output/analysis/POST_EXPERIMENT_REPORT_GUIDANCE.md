# Post-experiment implementation assessment and report guidance

Date: 24 October 2026

The completed work does not need to be discarded wholesale. Some findings affect documentation or reproducibility; others prevent particular results from answering the proposed comparison between implicit and explicit memory. The appropriate response before submission is to preserve the experimental record, disclose the affected implementations and conditions, and narrow conclusions to what was actually measured.

## Evidence and scope

- Reviewed the current environment and agent source, all three documents in `Phase Stories`, and the previously supplied research extract. The two excluded folders (`run_scripts` and `wandb_phase_3_baseline_exports`) and raw experiment histories were not inspected. Links from the stories into those folders were not followed.
- The stories document 24 Phase 3 baseline runs, tuning of QRM/Hazardous Delivery, DRQN/Sequential Colour, and PPO-RM/Deferred Maintenance, and Phase 4 sweeps of package count, readiness delay, and visual retention delay. These are documented configurations and reported results, not independently verified measurements.
- The user confirmed that maintenance experiments retained `max_stages=4`.
- Current repository HEAD: `1b8726050cf72a2cec369843f7bb8dd88617c21e`. The inspected environment, agent, test, story, and guide files had no local changes when checked. No cloud execution commit is recorded in the reviewed stories, so historical attribution assumes the executed code had the inspected behaviour. Source hashes are recorded in the diagnostic JSON.
- A standalone diagnostic script exercised seven groups of synthetic cases with the required interpreter, `D:\miniconda3\envs\temporal_gym\python.exe`. Custom layouts and state changes existed only in that process. No experiment implementation, trained model, or original result was changed, and no training was performed.

Diagnostic script: [post_experiment_audit.py](post_experiment_audit.py). Evidence: [post_experiment_audit_results.json](post_experiment_audit_results.json).

## Decision table

| Finding | Impact on completed experiments | Deadline response |
|---|---|---|
| 200-step smoke checks | Basic execution evidence only; they do not establish task solvability or scientific validity | Describe them as post-experiment smoke checks |
| Maintenance seeded reset | Reproducibility limitation; not evidence that every training run was unseeded | Disclose global NumPy RNG use and avoid claiming matched episode sequences |
| Inactive distractor/duration switches | No documented sweep of these features; would invalidate a claimed intervention using either switch | Remove those ablation claims and describe fixed hover timing |
| Hazardous lava proposition | Semantic label error; tested RM reward and terminal outcome are unchanged with the configured equal failure penalties | Disclose briefly; no retraining justified by this error alone |
| Sequential Colour QRM | Incorrect progress state and training reward; serious for the nine documented Phase 4 QRM conditions | Mark exploratory or exclude from the primary explicit-memory comparison |
| Maintenance PPO-RM | Readiness encoding does not track held codes and adds no new history information beyond the existing readiness input | Describe the implemented two-state wrapper; do not claim full explicit code memory |
| Maintenance delay horizon | Targets >=4 cannot activate readiness in the confirmed four-stage episodes | Remove those five conditions from memory-degradation inference; retain transparently as unreachable-readiness cases |

## 1. Smoke checks and Gymnasium checker results

All three registered environments executed 200 headless random-action steps with finite rewards and observations within their registered spaces. The core Gymnasium checker passed Hazardous Delivery and Sequential Colour; Deferred Maintenance failed the repeated seeded-reset comparison.

These are local, post-experiment observations. They do not demonstrate that all generated layouts are solvable, that all reward-machine labels are correct, or that a recurrent policy must outperform a reactive policy. Passing the checker is useful API evidence, not certification of an experimental design.

**Report wording:**

> Post-experiment validation confirmed basic reset/step and observation-space operation in all three environments. Targeted checks subsequently identified limitations in reproducibility, reward-machine event processing, and the feasible maintenance-delay horizon. These findings are accounted for when interpreting the experimental results.

## 2. Deferred Maintenance seeded reset: moderate reproducibility concern

The environment calls `super().reset(seed=seed)` but samples readiness and procedural layouts with the module-level `np.random`, rather than `self.np_random`. Thus `reset(seed=123)` does not independently reproduce a stage.

The probe produced different layout/readiness fingerprints for two successive `reset(seed=123)` calls. Reseeding global NumPy and reconstructing the environment reproduced the initial fingerprint exactly. This qualifies the earlier concern: it is not correct to infer that all completed training was unseeded or irreproducible.

The inspected SB3 installation's model seed helper seeds global NumPy, Python, Torch and the environment. The PPO, Recurrent PPO, and PPO-RM drivers pass a model seed. A controlled fresh process can therefore reproduce a random stream under appropriate conditions. However, environment generation shares its global RNG with other code: algorithm-dependent random consumption, evaluation, parallel threads, and checkpoint resumption can change subsequent layouts. The same numeric seed across algorithms does not establish paired, identical episode sequences. Cross-platform/cloud reproducibility was not established by the local probe.

**Deadline treatment:** retain the observations, disclose the RNG coupling, and report seed counts accurately. Present run-to-run variation if it already exists; do not manufacture uncertainty estimates from one run per setting. Do not claim exact replay from `env.reset(seed=...)` alone. A future implementation should isolate environment randomness using `self.np_random`; changing that now would create a different benchmark version.

**Report wording:**

> Training used explicit algorithm seeds. Deferred Maintenance generated stages and readiness targets using NumPy's global random generator, so its standalone reset seed did not fully determine episode generation. Comparisons therefore do not establish matched episode sequences across algorithms, and exact replay requires control of the wider random-number stream.

Sources: [environment generation](../../temporal_comp_gym/envs/deferred_maintenance/core.py), [SB3 reproducibility guidance](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#reproducibility), [Gymnasium reset API](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset).

## 3. Distractor and duration-matching switches: an unimplemented extension

`distractor_stations` and `duration_matching_mode` are assigned during initialization but never consulted by generation or service logic. With the global RNG controlled, toggling each flag produced identical layouts, observations, and rewards in the diagnostic. The source establishes that neither flag implements its advertised intervention.

`required_hover_duration` is a separate, active parameter; station service is timed using that fixed duration. Its default is two simulated seconds (120 increments of 1/60 second, subject to floating-point accumulation). This should not be described as dynamic duration matching. Ordinary unmatched stations can still occur relative to current inventory, but there is no implemented controlled distractor intervention.

The Phase Stories document package-count, readiness-delay and retention-delay sweeps. They do not document distractor or duration-matching ablations. Therefore these unused switches do not, on the available evidence, invalidate those documented manipulations. If a switch was nevertheless enabled in a particular experiment, it did not create the claimed treatment.

**Deadline treatment:** describe these as planned extensions, state that maintenance used fixed hover timing, and remove any claim to have measured the effect of enabling distractors or dynamic duration matching. No broad rerun is justified by an unused flag.

**Report wording:**

> The implemented maintenance task used a fixed hover-duration requirement. Controlled distractor-station and dynamic duration-matching interventions were outside the implemented experimental scope and are retained as future work.

## 4. Hazardous Delivery lava event: negligible learning impact in the tested RM

The wrapper identifies premature-goal entry by reward equality before checking lava. Both penalties default to -50, so a lava death can emit `reached_goal_prematurely` rather than `stepped_on_lava`/`died`.

This is a real semantic error, but its effect must not be overstated. For every active state of the three-package delivery RM, both the emitted label and the corrected lava labels produced exactly `(-1, -50, True)`: the same next RM state, reward, and terminal flag. More generally, this collision occurs when the two compared penalty values are equal, and the current RM treats those failure paths identically. The raw environment reward and termination are also unchanged.

**Deadline treatment:** no retraining is needed for this error alone. Do not use these propositions to report a reliable breakdown of lava deaths versus premature exits. Disclose label ambiguity if discussing event diagnostics. This equivalence does not certify other aspects of QRM or its counterfactual assumptions.

**Report wording:**

> A terminal-event labelling ambiguity merged lava deaths with premature goal entry when their penalties were equal. For the implemented delivery reward machine, both events had identical reward and terminal transitions; targeted checks therefore found no difference in the learning signal attributable to this ambiguity. Event-specific failure counts were not inferred from those labels.

Sources: [event wrapper](../../agents/rm_wrapper.py), [delivery RM](../../agents/qrm/train_qrm.py).

## 5. Sequential Colour QRM: serious limitation of the explicit-memory comparison

The Phase 3 baseline story contains DQN and DRQN on this environment; QRM was added for Phase 4. The documented affected scope is therefore the nine Phase 4 QRM delay conditions, rather than every visual experiment.

The wrapper detects `correct_wall` from any increase of `current_seq_idx`. That index also advances while colours are displayed. The probe confirmed false correct-wall events at memorisation steps 5 and 10 for a three-colour sequence, before navigation began.

The RM consequently starts navigation in an advanced state, can saturate early, and does not store the ordered colour identities. Correcting only the phase check would not turn this counter into an explicit representation of the colour sequence. With the default sparse environment reward, the RM also rewards each correct wall, never uses its stored completion reward, and pays for the false memorisation events.

In a controlled, correctly completed three-colour episode with no mistakes, the environment return was +10 while the RM return was +5: +2 during memorisation and +3 during navigation. The RM's own done flag remained false; the driver still terminates the episode through the environment's done flag. This is not merely a plot annotation error: QRM uses the RM rewards and transitions for its training targets.

The inspected QRM driver logs raw environment episode returns from an inner statistics wrapper. Consequently, its reported environmental return may still be numerically accurate; the training objective and supposed explicit memory are the compromised parts. The presence of false positive training rewards does not prove that the logged returns were inflated. Their empirical impact cannot be quantified without the excluded experiment records.

**Deadline treatment:** retain these measurements transparently as an exploratory prototype, or omit this arm from the primary comparison with correctly specified explicit memory. Keep the unaffected DQN/DRQN evidence, subject to its own evaluation limits. Do not interpret QRM's performance as a general success or failure of reward machines on visual retention. A plot correction or evaluation of old weights cannot retroactively repair the training objective; a faithful corrected comparison would require corrected training later.

**Report wording:**

> The visual QRM extension used a progress-counter abstraction that did not encode the memorised colour sequence. Post-experiment checks also identified phase-related event aliasing and a training reward that differed from the environment's sparse completion reward. These results are reported as exploratory implementation results and are excluded from conclusions about the relative effectiveness of fully specified explicit sequence memory.

The original QRM formulation exploits an explicit reward specification and its decomposition. Its guarantees should not be transferred to an incomplete event/reward representation or a partially observed neural approximation. See [Toro Icarte et al. (2018)](https://proceedings.mlr.press/v80/icarte18a.html).

## 6. Maintenance PPO-RM: readiness wrapper, not explicit maintenance-code memory

The maintenance RM has two states: waiting and ready. The underlying observation already contains the same readiness bit. Since readiness stays active after it is enabled, the appended one-hot is a redundant encoding of current information; it does not supply the missing inventory history.

A diagnostic constructed identical physical observations and RM states with held code 1 versus held code 2 at station 1. The complete augmented observations were identical, but the service rewards were +100 versus -10. This is a concrete counterexample to the claim that this RM state fully resolves the task's hidden code history.

An associated issue was confirmed while assessing rewards. The maintenance proposition wrapper uses `score_after - score_before` as its reward payload, rather than the raw step reward. Some branches add a cumulative step reward to score after score has already been incremented. In a synthetic step that both completed service and reached the final exit, the raw environment reward was +1100 and the score-derived RM reward was +1200. This demonstrates possible reward divergence; it does not establish how often that conjunction occurred in the real experiments. Many ordinary steps have matching values.

**Deadline treatment:** name the implemented method precisely, for example, "PPO with a two-state readiness wrapper". Its results can compare concrete implementations, but cannot establish that explicit memory of maintenance codes outperformed an LSTM. Report the score-derived reward as implemented. Avoid attributing a return advantage solely to memory: the extra encoding, reward computation, and differing tuned hyperparameters are possible explanations. Removing this causal claim is necessary even if every plotted raw return is correct.

**Report wording:**

> The continuous PPO-RM implementation augmented PPO with a two-state readiness machine. This state duplicated an existing observation variable and did not encode held maintenance codes; it therefore constituted a limited state-augmentation prototype rather than a complete explicit representation of the task's memory requirements. In addition, its training reward was derived from score increments, which can differ from the environment reward on compound transitions. Performance differences are consequently interpreted at the implementation level rather than attributed uniquely to explicit temporal memory.

Sources: [two-state RM and augmentation](../../agents/ppo_rm/train_ppo_rm.py), [score propositions](../../agents/rm_wrapper.py), [score/reward logic](../../temporal_comp_gym/envs/deferred_maintenance/core.py).

## 7. Confirmed four-stage horizon: high-impact restriction on Phase 4 interpretation

The user confirmed `max_stages=4`. Readiness is updated only when moving to another stage; the final-exit branch terminates before that update. Thus a target of k completed stages permits readiness during stage k+1 only when k<4.

The implementation uses an exclusive upper bound in `np.random.randint(low, high)`:

| Documented delay range | Actual sampled readiness target, in completed stages | Readiness reachable before final exit? |
|---|---|---|
| (1, 2) | 1 | Yes, during stage 2 |
| (2, 3) | 2 | Yes, during stage 3 |
| (3, 4) | 3 | Yes, during stage 4 |
| (4, 5) | 4 | No |
| (5, 6) | 5 | No |
| (6, 7) | 6 | No |
| (7, 8) | 7 | No |
| (8, 10) | 8 or 9 | No |

The diagnostic confirmed that readiness remained zero through final termination for targets 4 through 8. Targets 1 through 3 activated at the corresponding intermediate exits. These probes establish reachability of the readiness signal, not solvability of every generated task. Station availability, code capacity and layout feasibility remain separate concerns.

For the last five delay settings, a policy cannot perform valid maintenance servicing. An episode containing acquired codes cannot clear those obligations through successful service. Rare layouts with no surviving code obligations would not alter the fact that readiness and servicing were unreachable. Any low return or zero success in these conditions cannot identify an agent's memory failure threshold.

**Deadline treatment:** mark these five conditions (15 algorithm-condition rows in the documented sweep) as "readiness unreachable within four-stage horizon". Preserve their measured values in an appendix or visibly annotated plot, but exclude them from fitted memory-degradation curves, breaking-point estimates, and claims about which architecture retains information longest. The first three conditions are only candidates for that analysis; the RM limitations and unverified metrics still apply. Do not silently delete inconvenient rows or change `max_stages` and present the old data as corrected.

The story's "Max Delay Steps" label also needs correction. These parameters count completed stages, not environment time steps or seconds. A stage can take a variable number of actions. They therefore do not establish a linear increase in elapsed memory time.

**Report wording:**

> The maintenance delay sweep used a fixed four-stage episode horizon. Subsequent validation established that readiness targets of four or more completed stages could not activate before episode termination. These conditions are retained as unreachable-readiness cases and excluded from inference about memory degradation. For the remaining conditions, the delay parameter denotes stage transitions rather than a fixed number of simulation steps.

## 8. Claims in the Phase Stories that should not be copied directly

The stories contain confident interpretations that exceed the recorded evidence. They are useful execution summaries, but should not substitute for checking the metric definition and the supporting observations.

- **Zero return is not a success-rate measurement.** A policy that avoids both reward and penalties until timeout can return zero without completing a task. No-episode records or summary fallbacks are also possible explanations of zeros; neither was verified here. Describe an unresolved zero only as the recorded metric until its origin is checked.
- **Positive return alone does not demonstrate full completion.** In default Hazardous Delivery, success requires every delivery plus goal entry; at eight packages the undamaged full return is 8 x 10 + 50 = 130. A reported +1 alone does not prove eight deliveries. It also does not rule out some successful episodes if it is an average. Task-success claims require an episode-level success definition.
- **Negative visual return does not demonstrate retained memory.** The documented -11 at delay 20 cannot on its own establish recall; incorrect attempts also generate negative rewards. The reported +8 at delay 12 is promising conditional evidence of completion if it is an actual episode return under the stated sparse reward, but should not be promoted into a population success rate without the metric and repetitions.
- **Do not infer mechanisms such as vanishing gradients, catastrophic forgetting, combinatorial state explosion or unavoidable hazard crossings from a reward curve alone.** Use "performance decreased under the tested configuration" unless diagnostics support the mechanism.
- **Training-seed uncertainty and episode uncertainty are different.** Phase 3 documents seeds 1-3. The Phase 4 registry documents one training seed per algorithm-condition entry, including differing seeds across visual algorithms. If that registry is complete, do not describe Phase 4 as replicated across three training seeds. Repeated evaluation episodes of one policy do not provide three independent training runs.
- **Tuning was selective.** The tuning story targets one representative per environment. Do not state that every competing algorithm received an equal tuning budget unless other evidence establishes that. Attribute comparisons to the tested configurations.
- **Package count is a task-capacity and spatial-density manipulation as well as a temporal one.** It does not isolate a linear delay horizon. Use separate labels for package count, completed-stage readiness target and blank-frame retention delay.
- **Memorisation success should be empirical, not asserted from the algorithm name.** At default Hazardous Delivery settings, health and delivered count are visible. The local map and carried inventory remain limited, but the setup does not justify describing every feedforward failure as a theorem about hidden delivery-count memory.

Suggested replacements:

| Avoid | Use |
|---|---|
| "The experiments definitively prove memoryless agents cannot solve these environments" | "Under the tested configurations and budgets, the reactive baselines showed the reported performance limitations." |
| "QRM solved eight-package delivery because its return was +1" | "QRM recorded the reported return at eight packages; full-task success requires a separately verified completion metric." |
| "PPO-RM proves explicit memory is superior in continuous control" | "The tested readiness-wrapper PPO configuration achieved the reported return; the experiment does not isolate explicit maintenance-code memory." |
| "The LSTM's memory breaks at delay k" | "Performance under this implementation changed at the tested delay; the available design does not isolate a general memory-capacity threshold." |

## 9. Submission priorities and optional follow-up

1. Preserve the executed code and raw results as the experimental version. Keep this audit separate. A later patch should be versioned as a correction, not presented as what produced the existing results.
2. Correct the methodology to describe actual observation visibility, fixed maintenance timing, actual RM state contents, training-reward definitions, tuning coverage and seed counts.
3. Annotate the five unreachable maintenance conditions and remove them from memory-breaking-point inference.
4. Separate the visual QRM prototype from the primary explicit-memory conclusions; qualify the maintenance PPO-RM comparison as above.
5. Check any available, already produced success and episode-completion metrics before using the stories' numeric interpretations. This audit did not inspect the excluded result files. Missing evidence must be labelled as unavailable rather than inferred from zero or positive return.
6. Add a clear limitations paragraph and frame the contribution as a multimodal benchmark implementation with preliminary, implementation-specific empirical findings. The broader hypothesis remains a motivation rather than a universally resolved claim.

If time permits, an independently seeded, common-metric evaluation of already trained checkpoints could improve behavioural evidence without retraining. It must use a clear success criterion, a common time limit, raw environment returns and recorded seeds, and retain the input interface expected by each trained model. Such evaluation can assess those frozen policies; it cannot repair training under an incorrect RM, establish sample efficiency from new final-policy scores, or create missing independent training replications. Fixing the RM and training again belongs to future validation if the deadline does not allow it.

### Combined limitations paragraph

> This study provides a multimodal benchmark implementation and preliminary comparisons of selected reactive, recurrent and reward-machine-based implementations. Post-experiment validation identified limitations that restrict interpretation. Deferred Maintenance coupled environment generation to NumPy's global random stream, and readiness targets at or beyond the fixed four-stage horizon could not activate before termination; these conditions were excluded from memory-degradation inference. The continuous PPO-RM used a two-state readiness abstraction that did not encode held maintenance codes, while the visual QRM extension contained phase-related event aliasing and a reward specification that differed from the environment objective. Results from these prototypes are therefore interpreted as implementation-specific observations rather than evidence of general architectural superiority or failure. Controlled distractor and dynamic duration-matching studies, fully specified memory abstractions and broader independent validation remain future work.

### A defensible conclusion

> The project demonstrates the practical construction of temporally structured tasks across discrete, continuous and visual modalities, together with integrated learning agents and configurable experimental conditions. The completed experiments provide preliminary evidence about the behaviour of the tested implementations. They do not establish a universal ranking of implicit and explicit memory or precise architecture-level memory breaking points. The implementation audit identifies concrete requirements for a more controlled follow-up comparison, including faithful event abstractions, equivalent evaluation objectives and verified task feasibility.
