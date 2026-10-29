# T2 local validation — 2026-10-29

Implementation and validation were performed on branch `new_experiments`, starting at `a0531727f44ede8134dc2d6732af4e71ddbc3011`, in the requested Windows checkout. Python was `D:\miniconda3\envs\temporal_gym\python.exe`, with `-B`; subprocesses also inherited `PYTHONDONTWRITEBYTECODE=1`. No packages were installed, machines rented, cloud jobs launched, credentials supplied or changes pushed.

## Results

- The registered full study is **154 runs**: 88 candidates and 66 confirmations, with role counts **A 27 / B 74 / C 53**. `generate_trials.py --check` passes. Plan hash: `e160f2cf0fdacdca89218135287bafc9aefcbcd21d695582365da2114a857ddd`.
- The local study completed **88/88 diagnostic runs** across all 11 cells: 22 screening runs plus 66 confirmations. Each uses 64 training steps, short episodes and real updates; all study W&B runs were offline. The smoke role counts are A 12 / B 46 / C 30.
- Strict verification passed for each trial's config/plan identity, exact budget, updates and finite losses, greedy/noisy evaluation cadence and episode seeds, and terminal validation. Confirmation records match the frozen finalist assignment. Selection produced all 11 entries without an incomplete-data override; `usable_for_final_comparison` is **false** because this is smoke.
- All selected smoke success AUCs were zero. These diagnostic budgets and shortened episodes provide no evidence about scientific learning performance. Return or candidate-ID tie-breaks exercised the all-zero-success selection path; unit tests additionally cover nonzero progress and success ranking.
- Four checkpoint directory replacements encountered Windows access errors: one screening attempt was restarted, and three confirmation attempts resumed. The original failed screening attempt and three pre-resume failure-status JSONs are retained. Final evidence has 89 attempt directories, 88 complete attempts and one superseded failed attempt. No failure was converted to a zero score.
- A separate injected failure at step 48 resumed from step 32 and completed with four DQN updates. Continuation skipped verified completed trials. Git Bash wrapper dry-run passed; native Linux execution remains a cloud check.
- All **440** pre-change v2/v3 baseline hashes (full/smoke, seeds 1–10) match the regression fixture. `baseline_plan write --check` passes and the frozen v2 manifest remains **55/55**. Baseline result/validation files were not modified.

## Tests and evidence

The final full suite passed: **107 tests in 483.75 seconds**, using `python -B -m pytest -p no:cacheprovider -q` with a fresh `--basetemp` and one CPU thread. An additional targeted preflight path-length check passed after adjusting the SDK cache reserve to accommodate temporary preflight roots. Tuning tests cover deterministic generation and spaces, budget divisibility, fixed task/discount/secondary evaluation, seed separation, AUC and lexicographic ties, unavailable results, explicit incomplete-data and duplicate-attempt choices, confirmation ranking, path limits, budget guards, launcher failures/continuation, checkpoint retries and CSV/JSON-only collection.

An initial full-suite run exposed the tuning terminal seed block being treated as a separate top-level million-seed block; it is now correctly a sub-block of tuning's existing range. Git metadata and Git Bash temporary-directory checks initially encountered sandbox ownership restrictions. Successful suite execution uses a fresh temporary directory and process-local Git `safe.directory`, with no global Git configuration changes. Thirty-one early smoke metadata files lack a Git revision because of that ownership restriction; their original records are preserved. Later metadata records the verified starting revision and dirty implementation checkout. This evidence is engineering validation, not a preregistered production dataset.

[`smoke_evidence.zip`](smoke_evidence.zip) contains **1,286 CSV/JSON members**, including all collected attempts, frozen finalists, selection evidence and the exact trial list. It is 735,954 bytes and contains no checkpoints, replay buffers or W&B cache. Its longest unpacked path is **223 characters** using the specified 92-character checkout prefix. Raw smoke files and checkpoints remain local under `results/v3/smoke/`; they are ignored by Git. The rendered selection report is there as `selection_report.md`; its machine-readable evidence is in the bundle.

## Open cloud checks

- CUDA/RTX 3090 learning and restart checks, GPU/VRAM and cumulative disk capacity.
- Online W&B login/connectivity and resume behaviour, using researcher-managed authentication.
- Native Linux Bash launch, interruption, continuation, and three-machine archive transfer/merge.
- Actual v3 throughput, evaluation overhead, checkpoint sizes and billed-time projections before committing the remaining compute allowance.

The launch guide documents the 100-hour/~$30 compute guard and its fixed role shares. The launcher can stop work, but cannot stop rental billing: the researcher must stop idle machines. No final comparison or stress-test execution was implemented.
