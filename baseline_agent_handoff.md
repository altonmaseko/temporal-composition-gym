# Baseline experiment handoff for new_experiments

Use the shared context first, then send the phase prompts in order to the same agent. Each phase has a concrete deliverable. Full training is a later, user-operated step; these prompts do not authorize renting infrastructure or starting production cloud runs.

## Shared context to give the agent

```text
I am rebuilding the experimental pipeline for Temporal Composition Gym on the existing branch new_experiments. This is the exact branch name; an earlier message misspelled it as new_expriments.

Protect the existing main checkout. Work in a separate clone or Git worktree already checked out on new_experiments. Before any edit, verify the current branch and checkout path. If you are in the preserved main checkout, inspect the branch read-only with git show/git ls-tree and report the checkout problem instead of switching that directory. Do not edit, reset, clean, or switch the preserved main checkout. Do not overwrite my existing changes.

On my Windows computer, all Python commands must use D:\miniconda3\envs\temporal_gym\python.exe, and package installation must use that interpreter with -m pip. Follow AGENTS.md. Cloud Linux launchers must use the explicitly selected Python from the provisioned cloud environment; do not hardcode my Windows path into Linux scripts or assume arbitrary system Python is acceptable. Python subprocesses should use sys.executable. Make the shell interpreter selection explicit and verify it during preflight.

My research questions are:
1. How do explicit automaton-based reward representations compare with implicit memory-augmented networks in sample efficiency?
2. How does performance degrade as temporal delay increases?

We are implementing and validating BASELINES ONLY now. Do not create or revise hyperparameter tuning or stress-test drivers in this work. Design the baseline interfaces so later tuning and stress work can reuse them, but keep those later experiments separate. Do not select parameters from historical result winners as though those winners were validated.

The branch contains implementations and scripts, but I removed prior results. The old results and Phase Stories describe pilot work and are not authoritative evidence that the old experiments were valid. Their useful context is summarized below. Missing results do not prevent code corrections or controlled local validation.

Current baseline paths:
- run_scripts/baseline/run_memoryless_baselines.sh
- run_scripts/baseline/run_implicit_memory_baselines.sh
- run_scripts/baseline/run_explicit_memory_baselines.sh
- run_scripts/baseline/rerun_drqn_image_and_ppo_rm_continuous.sh

These main launchers call:
- agents/train_dqn.py
- agents/train_ppo.py
- agents/train_recurrent_ppo.py
- agents/drqn/train_drqn.py
- agents/qrm/train_qrm.py
- agents/ppo_rm/train_ppo_rm.py
- agents/rm_wrapper.py

All three environments are under temporal_comp_gym/envs/. Necessary corrections to environments, observation wrappers, reward machines, tests, dependencies and shared training helpers are in scope. Fixing only the shell files would leave the comparisons scientifically invalid. Preserve the intended three tasks and document material changes to their semantics.

Cloud layout: three independently rented GPUs/machines. The memoryless launcher runs on one machine, the implicit-memory launcher on a second, and the explicit-memory launcher on a third. Each machine runs only its assigned launcher. Default to one training process at a time per machine, proceeding through its environments and seeds sequentially. Do not launch the old six-job parallel remediation script. GPU models, VRAM and RAM may differ; inspect capabilities and save hardware metadata rather than assuming an RTX model or memory capacity. Sample-efficiency comparisons use matched environment interactions, not matched wall-clock time. Report compute differences separately.

Keep using W&B for every production training run. Also preserve complete local numeric records so analysis does not depend on sampled W&B history exports or stdout parsing. Place baseline results under run_scripts/baseline/results/, with isolated experiment/run subdirectories. Keep checkpoints/logs/metadata either there or under a clearly documented baseline-specific artifact root referenced from each result directory. Tuning and stress artifacts must never share baseline checkpoints.

Required comparison structure:
- Memoryless: DQN for Hazardous Delivery and Sequential Colour; PPO for Deferred Maintenance.
- Implicit: DRQN for Hazardous Delivery and Sequential Colour; Recurrent PPO for Deferred Maintenance.
- Explicit: QRM for Hazardous Delivery; PPO-RM for Deferred Maintenance; a scientifically valid explicit baseline for Sequential Colour must be defined and validated before claiming three-way coverage of all three environments. The current explicit shell script omits the colour task. Its current colour QRM code is not a valid solution to that gap.

Do not silently omit the explicit colour comparison, implement a progress-only placeholder, or call an alternative method QRM without explaining how it relates to QRM. Establish a defensible design in Phase 1. If that design genuinely requires a material change in research scope, present the concrete issue and options before dependent implementation. Resolve routine engineering choices yourself.

Known findings from the previous audit, to independently verify against the code:
1. SB3 checkpoint lookup and names omit seed and task configuration and automatically resume. Pilot seed-2/seed-3 runs often had empty histories or began near step 900,000. Custom checkpoint lookup later gained seed isolation, but still lacks a full experiment identity. Fresh runs must not inherit weights, optimizer state, replay or counters from other experiments.
2. The old remediation script omitted timestep arguments: image DRQN requested only 100k instead of 3M, and maintenance PPO-RM 100k instead of 2M. Image DRQN pilot records stopped around 22k-29k steps without recorded TD losses. The main three launchers must cover the whole matrix with explicit budgets, resource-safe implementations, and verified learning updates. No separate remediation script should be required.
3. Current DRQN flattens images into an MLP rather than using a visual encoder. Its replay capacity is measured in episodes and can imply enormous memory usage. It clears recurrent hidden state on random actions, trains random windows from zero hidden state without burn-in, and treats truncation like terminal failure in targets. Review these mechanisms carefully rather than merely lowering timesteps until it runs.
4. SequentialColourRMWrapper emits correct_wall when current_seq_idx advances during memorisation. Its RM advances a counter, does not retain the flashed target sequence, and produces per-wall rewards even under sparse environment rewards. Memorisation cues, genuine wall interactions, and terminal completion must be distinguished. A symbolic agent must derive its memory from the permitted cue/event stream, not read hidden target answers during navigation. Ensure counterfactual QRM updates are valid for the chosen representation.
5. DeliveryRewardMachine tracks delivered count only. The default base observation already exposes delivered count and health. Maintenance's two-state RM tracks readiness, which is already visible in the base observation, but does not retain held maintenance codes. Explicit states must actually encode the relevant task history. Specify the information and task-structure advantage provided to explicit agents and the observation available to other agents. Do not claim RM augmentation alone makes the locally observed navigation process fully observable.
6. Hazardous Delivery infers terminal propositions by comparing scalar penalties. Lava and premature-goal penalties are both -50, so lava is labeled as premature goal. Emit explicit events from the transition semantics and distinguish reward/termination causes.
7. Deferred Maintenance uses global NumPy randomness. Repeated reset(seed=...) failed Gymnasium determinism checks. The score accumulator sometimes adds the whole accumulated step reward again, while the RM obtains reward from score differences; check reward equivalence and remove unintended double counting. Its macro waypoint is currently [0,0]. Some advertised toggles are stored but do not affect dynamics. Validate the baseline configuration and avoid claiming unimplemented capabilities.
8. Maintenance defaults to four stages. Readiness thresholds at stage 4 or later cannot activate before terminal exit; episode horizons must also allow collection, retention, navigation and the required service duration. Do not choose a baseline where success is structurally impossible.
9. Current tests are visual rollout demos rather than semantic regression tests. Add meaningful automated checks for the scientific assumptions and these defects, including deterministic seeds, task success/failure, event timing, rewards, and episode boundaries.
10. Pilot training used unequal budgets, different return definitions and incomplete logging. SB3 rolling training means and custom agents' individual training returns are not a common evaluation metric. Never use the last training episode as the headline algorithm score.
11. Old stress scripts converted parse failures into zero and ignored subprocess exit codes. Avoid that pattern in the baseline pipeline. Missing/failed/incomplete results must be explicit and cannot masquerade as measured zero performance.
12. QRM/DRQN tuning selected the globally newest TensorBoard folder while running multiple workers, so trial objectives could belong to other trials. Do not reuse that log-selection architecture in shared baseline helpers or later handoff interfaces.

Pilot evidence includes occasional QRM +80 episodes on the three-package task and occasional colour-sequence completion. These show useful observations, not validated algorithm rankings, stable convergence, or memory breaking points. Do not force results to favor explicit memory or to make feedforward policies fail.

The existing full-budget launchers request 1M steps for delivery, 3M for colour, and 2M for maintenance. Treat these as explicit starting budgets to review for feasibility, not as proven sufficient budgets. Match budgets across algorithms within each environment. SB3 rollout rounding should be recorded and handled consistently. Recommend a documented seed set and justify it; five independent training seeds is a reasonable initial baseline plan, with provision to expand based on variability. Do not treat any seed count as an automatic statistical guarantee.

Use the staged prompt below for the work currently authorized. Do not run full experiments, rent GPUs, access my W&B account through credentials, or start cloud jobs during the implementation/preflight stages. Short local/offline validation is authorized. Preserve credentials and do not print or commit them.
```

## Prompt 1: audit and freeze the baseline protocol

```text
Using the shared context, inspect new_experiments and produce a concrete baseline correction plan and experimental protocol. This phase is read-only with respect to project implementation. Do not start training or change source files.

Verify the actual branch, relevant instructions, three baseline launchers, six trainers, RM wrappers, observation spaces, reward/termination semantics and replay mechanisms. Distinguish confirmed code defects from historical hypotheses.

Specify the full environment/algorithm matrix, common task configurations, observation/information contracts, valid RM state and event definitions, task-success criteria, episode limits, interaction budgets, independent training seeds, held-out evaluation episode seeds, evaluation cadence, policy action-selection rule, final-checkpoint rule, and local/W&B output schema. Evaluation seeds must not be the training stream, and later tuning/test seed sets must be separable.

Address the missing explicit colour baseline explicitly. Explain how the representation retains observed colours without hidden-answer leakage, handles per-episode random sequences, supports learning and valid QRM counterfactual updates if QRM is used, and avoids impractical state/replay growth. Also explain what task memory the delivery and maintenance machines retain beyond information already visible in the base observation. A progress counter or readiness bit is insufficient.

Specify environment solvability and oracle/fully observed controls, and distinguish the main non-Markovian condition from optional observability ablations. Do not make a hidden-variable choice that confounds all navigation and memory claims without documenting it.

Describe resource estimates for image replay/encoders and the three-machine launch plan. Identify the exact files to change and a testable acceptance criterion for each major correction. Recommend practical engineering defaults rather than asking me about every implementation choice. Flag only genuinely consequential research-design choices that need my decision.

Return a concise, implementation-ready plan with remaining uncertainties. Stop before implementation so I can supply Prompt 2 once the protocol is agreed.
```

## Prompt 2: correct task semantics and memory implementations

```text
Implement the baseline protocol agreed in Prompt 1 on new_experiments. Verify you are in its separate working directory before editing. Focus this phase on environments, observations, reward machines, the correctness of the agents' memory mechanisms, and meaningful automated tests. Necessary shared helpers are allowed. Do not start production training or implement tuning/stress experiments.

Correct the confirmed semantic defects: seeded RNG use; explicit transition events; success/failure and termination/truncation information; raw reward and RM reward consistency; feasible maintenance readiness/service rules; and a valid explicit task-memory representation for each selected task. Implement the agreed colour representation rather than a progress-only placeholder. Validate any counterfactual replay assumptions. Keep current research tasks recognizable and document behavior changes.

Correct DRQN hidden-state handling and sequence training, including history updates during exploration, appropriate window context/burn-in or an equivalent justified approach, padding masks, and terminal-versus-time-limit target handling. Provide a practical visual feature extractor and memory-bounded image replay with justified preprocessing. Avoid losing colour information through unsuitable grayscale preprocessing. Avoid sharing mutable RNG/evaluation state with training. Check that all algorithms learn from the intended transitions and that different replay-capacity units are explicit.

Add automated semantic tests, not just random visual loops. Include cue-versus-wall event timing, repeated colours where supported, actual sequence completion, collection/delivery/maintenance history, lava versus premature goal, fatal damage, healing, raw/RM reward agreement, no leaked navigation target, deterministic resets and rollouts, truncation handling, and successful completion at the chosen baseline settings. Use tiny controlled layouts or scripted trajectories where useful. Demonstrate the tasks are solvable independently of successful neural training.

Run appropriate checks with the prescribed Windows interpreter. Use offline/no-network logging for local validation. Summarize what changed, tests run, and any unresolved scientific limitations. Do not claim experiment readiness if the explicit colour design or environment solvability is still unresolved. Leave hyperparameter tuning and stress scripts untouched.
```

## Prompt 3: unify baseline training, W&B, evaluation and launchers

```text
Continue on new_experiments using the accepted protocol and corrected implementations. Make the six training entry points and the three baseline .sh launchers a complete, reproducible baseline pipeline. Add shared code where it removes duplicated behavior, while keeping the public launchers straightforward. Do not implement HPO or stress sweeps, and do not run full training.

Requirements:
1. The three launchers cover the agreed complete comparison matrix. Image DRQN and maintenance PPO-RM are ordinary runs in the implicit/explicit launcher, with explicit full budgets. A separate remediation launcher is unnecessary. Keep rerun_drqn_image_and_ppo_rm_continuous.sh as an explicitly deprecated compatibility entry or replace it with a clear message pointing to the main launchers; do not execute it as part of baseline work.
2. Resolve repository root correctly from run_scripts/baseline/, independent of caller working directory. Use robust shell error handling, properly quoted arguments, a selected cloud interpreter, visible process failures, and per-run stdout/stderr logs. If a batch continues after a failure, record it explicitly and make the overall script exit nonzero. Never print a success summary when jobs failed.
3. Default to one sequential training process per assigned GPU/machine. Support device selection and record the actual selected device. Do not silently call a CPU fallback a GPU run. Do not assume every operation is GPU-bound or that GPU models are identical. Provide thread controls and a resource preflight without automatically installing into arbitrary environments.
4. Use one resolved configuration per run containing environment kwargs, model/optimizer/replay settings, seeds, episode/evaluation limits and budgets. Log the complete configuration to both disk and W&B. CLI/config overrides must be unambiguous. Do not depend on undocumented shell environment variables whose values are missing from metadata.
5. Use unique, collision-resistant experiment/run IDs and config hashes. Include phase, environment, algorithm, seed and relevant configuration in identity. All outputs go to isolated baseline run directories under run_scripts/baseline/results/ or an explicitly linked baseline artifact root. Two independently rented machines must produce outputs that can later be merged without overwriting each other.
6. Fresh training is the default. Never scan a shared models directory and silently resume. Explicit resume must verify compatible configuration, original seed, run identity and state; report restored and newly executed steps separately. Save the algorithm state required for a documented restart, including replay for off-policy methods if claiming continuation. Restore RNG/target/optimizer/counters/schedules as appropriate. If environment/recurrent state cannot be restored exactly, document the episode-boundary restart semantics and test them. Do not advertise exact continuation without evidence.
7. Separate training and evaluation environments and random streams. Implement periodic frozen-policy evaluation and final evaluation using the agreed action-selection rule. Carry/reset recurrent hidden state correctly during evaluation. Log raw environment returns and explicit task success, not just RM learning rewards. Record complete per-episode evaluation data and termination/truncation causes. Evaluation must not update the policy or training normalizers/replay, and must not advance training RNG streams.
8. Use consistent metric names and a real environment-interaction x-axis in W&B. Separate training metrics from eval/return_mean, eval/success_rate, episode counts, dispersion and final evaluation. Log learning/optimizer updates, exploration, actual timesteps and resource usage to distinguish collection-only runs from learned policies. Ensure W&B initialization, metric logging, finish/flush and failure status work for SB3 and custom agents. Do not assume WandbCallback automatically uploads all TensorBoard scalars.
9. Preserve full local training/evaluation numeric records. Save configuration, git revision, relevant dirty-code information without secrets, package/interpreter versions, hardware metadata, W&B ID/URL, checkpoints and status. Do not parse human stdout or sampled history to construct official scores. Missing/failed/incomplete results must remain distinct from zero. Only mark a run complete after verified budget completion and final evaluation.
10. Provide a manifest listing expected and completed environment/algorithm/seed combinations, a validated completion check, and documented commands to collect/merge the three machine outputs. Target the 'new_experiments' W&B project explicitly, loading the project configuration and API key from the local .env file. Do not hardcode credentials or default to the old pilot project.

Use common budgets within each environment and a documented seed list, with configurable smoke/full modes. Handle SB3 rollout-budget rounding explicitly. Record episode-length differences and learning-update counts. A reduced smoke mode must label outputs as smoke and must never be mixed into production results.

Deliver the corrected launchers/trainers, supporting tests, concise launch/collection documentation, and a machine-readable resolved baseline plan. Verify shell syntax where a compatible shell is available and Python behavior using the prescribed local interpreter. Report checks you could not run. Do not launch the full GPU jobs.
```

## Prompt 4: run the preflight and prepare the three-machine launch

```text
Validate the corrected baseline pipeline end to end before I rent/start the three production machines. Remain on the isolated new_experiments checkout. Run short, explicitly labeled local/offline checks only. Do not access my W&B credentials, rent infrastructure, or launch full cloud training.

Execute at least one short end-to-end run for every agreed environment/algorithm combination. Make each run long enough to trigger actual optimizer updates, at least one evaluation and a checkpoint save/load; use explicit smoke settings to shorten warm-up or schedules, never silently modify production settings. Record which settings differ. A run that merely instantiates the policy or fills replay is not a passing training test.

Check image memory use and the projected full-run replay allocation; finite observations, losses and returns; recurrent evaluation state; correct events and task success; fresh-seed isolation; configuration validation; checkpoint/resume semantics; artifact completeness; actual budget accounting; and strict separation of evaluation from learning. Run a two-seed/two-config collision check to show the second run cannot silently inherit the first. Exercise a deliberate failure and a missing-metric case to verify neither becomes success/zero. Validate offline W&B records without treating them as proof of online upload.

Provide a lightweight cloud preflight command for each rented machine. It should validate the selected interpreter, dependencies, CUDA/device, RAM/VRAM, a short learning run and online W&B logging once I run it with credentials. Online W&B upload and real cloud-GPU operation remain unverified until that cloud preflight succeeds. It must not consume the full training budget.

Produce three exact production commands: one for the memoryless machine, one for implicit memory, and one for explicit memory. State the shared code/config revision, expected run matrix and count, budgets, seed list, W&B grouping, output locations, restart procedure, failure checks, and how I collect outputs from all machines. Mark production readiness only if the agreed comparisons and local checks are implemented; list any remaining cloud-specific checks separately.

Do not implement hyperparameter tuning or stress-test scripts. End with the baseline collection contract: the precise artifacts and fields the later analysis agent will receive. The next phase after production runs is analysis and validity checks; tuning design will then use those diagnostics.
```

## After baseline collection

Use a separate later prompt to reconcile expected runs, verify actual new-interaction budgets and independent seeds, inspect learning and evaluation records, and summarize variability and failures before drawing rankings. Then design comparable hyperparameter searches with their own directories, training seeds and validation seeds. Final algorithm comparisons should use selected configurations on fresh testing seeds. Stress testing follows the corrected and validated task semantics.

Initial default-setting baselines can inform tuning; HPO code does not logically require baseline results to exist. The reason to defer task-specific HPO now is to keep this work focused and let diagnostics inform the searches. The initial baselines are not automatically the final tuned evidence for RQ1.
