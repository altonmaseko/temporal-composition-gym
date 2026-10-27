# Automated Test Suites and Verification Scripts

The `tests/` directory contains the automated `pytest` suite for protocol v2. Unlike the earlier smoke scripts, which only ran random actions and printed spaces, these tests **assert specific behaviour**: the task rules and events of each environment, the reward machines, the custom learners, seeding, and the baseline training pipeline (run directories, failure handling, resume, launchers).

> **Changed from the pilot.** The old `tests/test_<env>.py` random-rollout scripts that opened a window are now display demos in [`tests/visual/`](../tests/visual/) (`demo_hazardous_delivery.py`, `demo_deferred_maintenance.py`, `demo_sequential_colour.py`). Their names start with `demo_`, so pytest does not collect them. Run one directly to watch random play; adjust its `time.sleep(x)` to change the speed.

---

## Running the Tests

On the development PC, always use the project interpreter (see `AGENTS.md`). Run from the repository root:

```bash
D:\miniconda3\envs\temporal_gym\python.exe -m pytest
```

Useful variations:

```bash
D:\miniconda3\envs\temporal_gym\python.exe -m pytest -m "not slow"                 # skip the 7 slow tests
D:\miniconda3\envs\temporal_gym\python.exe -m pytest tests/test_hazardous_delivery.py
D:\miniconda3\envs\temporal_gym\python.exe -m pytest -k counterfactual -v           # tests whose name matches
```

On a Linux experiment machine, use that machine's interpreter (for example `~/tcg/bin/python -m pytest`).

### Configuration (`pytest.ini`, `tests/conftest.py`)

*   `testpaths = tests` and `pythonpath = .`, so the tests import the code in this checkout without an install. `test_code_is_imported_from_this_checkout` checks this.
*   The `slow` marker labels longer checks: oracle success rates, resume, learning on a toy task and the `--continue-batch` restart.
*   Temporary run directories are kept only for failing tests (`tmp_path_retention_policy = failed`).
*   `UserWarning` and `DeprecationWarning` are filtered.
*   `conftest.py` sets `SDL_VIDEODRIVER=dummy` (no windows open) and `WANDB_MODE=disabled`. Pipeline tests use `--wandb-mode offline` or `disabled`, so the suite never contacts W&B.
*   All training runs inside the tests write to pytest's temporary directories, never to `run_scripts/baseline/results/`.
*   The launcher tests need a POSIX `bash`; on Windows they use Git Bash if it is installed and are skipped otherwise.

---

## What Each Test File Checks

### 1. Hazardous Delivery (`test_hazardous_delivery.py`)
> 📁 [`tests/test_hazardous_delivery.py`](../tests/test_hazardous_delivery.py)

Most tests build the main-condition environment, clear the grid and place objects by hand, so each rule is checked in isolation.

| Test | Checks |
|---|---|
| `test_main_condition_hides_delivered_count` | The observation has 8 values (no delivered count) and health is the last value. |
| `test_lava_is_not_premature_goal` | Lava gives event `lava`, −50 and cause `lava`; an early goal gives `goal_contact`, −50 and cause `premature_goal`. |
| `test_scripted_success_rewards_and_history` | A scripted run of three pickups and deliveries then the goal gives rewards 0, 10, 0, 10, 0, 10, 50, the right events and hidden counts, and cause `success`. |
| `test_capacity_and_empty_handed_contacts` | A second package while full, or a destination while empty-handed, gives only a contact event, no reward, and the object stays. |
| `test_damage_fatal_damage_and_healing` | Damage costs −1 and 1 health; fatal damage gives −51 and `damage_death`; a health node restores full health and disappears. |
| `test_truncation_is_not_termination` | Reaching `max_steps` gives `truncated=True`, `terminated=False`, cause `timeout`. |
| `test_seeded_reset_and_rollout_are_deterministic` | The same seed gives the same layout and rollout; a different seed gives a different layout. |
| `test_every_generated_layout_is_solvable` | Seeds 0–999 all produce layouts where every objective is reachable without lava. |
| `test_unsupported_task_dependencies_raise` | `task_dependencies=True` raises `NotImplementedError`. |
| `test_oracle_solves_final_eval_seeds` (slow) | The scripted oracle succeeds on all 200 final-evaluation seeds in fewer than 500 steps. |

### 2. Deferred Maintenance (`test_deferred_maintenance.py`)
> 📁 [`tests/test_deferred_maintenance.py`](../tests/test_deferred_maintenance.py)

| Test | Checks |
|---|---|
| `test_gymnasium_env_checker_passes` | Gymnasium's `check_env` passes (including its determinism checks). |
| `test_seeded_layout_and_rollout_are_deterministic` | The same seed gives the same layout, schedule and rollout; a different seed differs. |
| `test_score_is_the_running_sum_of_rewards` | `score` equals the running sum of step rewards. |
| `test_schedule_is_solvable_by_construction` | Over 300 seeds: readiness stage 1–2, distinct issued codes that fit the capacity, all 3 station types (distractors on), exactly one station per type in every stage, codes only in their scheduled stages. |
| `test_invalid_or_unimplemented_configurations_raise` | Invalid readiness ranges raise `ValueError`; `duration_matching_mode=True` raises `NotImplementedError`. |
| `test_macro_waypoint_points_to_next_path_tile` | The scaled waypoint equals the offset to the next path tile, in cells. |
| `test_code_collection_event_and_tile_conversion` | Touching a code gives `code_collected(k)`, adds it to the held codes, turns the tile into path, and shows the floor code. |
| `test_premature_success_and_wrong_service` | Hovering 120 steps gives one `service_attempt`: −10 before readiness, +100 with the held code when ready, −10 with a code not held. |
| `test_exit_outcomes` | Exiting with an uncollected code gives −1000 and `uncollected_codes_at_exit`; the final exit gives +1000 `success` or −1000 `held_codes_at_final_exit`. |
| `test_truncation_at_episode_limit` | Reaching `max_steps` truncates with cause `timeout`. |
| `test_readiness_switches_on_when_entering_the_readiness_stage` | Driven by the oracle: `readiness_on` arrives with the exit of stage r−1, the oracle never mis-services, and the episode succeeds. |
| `test_oracle_success_and_horizon` (slow) | Oracle success ≥ 95% on 40 final-evaluation seeds, with the 95th-percentile length at most half the step limit. |

### 3. Sequential Colour (`test_sequential_colour.py`)
> 📁 [`tests/test_sequential_colour.py`](../tests/test_sequential_colour.py)

| Test | Checks |
|---|---|
| `test_cue_delay_and_nav_event_timing` | The first `cue` is in the reset info; later `cue`, `delay_start` and `nav_start` events arrive on exactly the right steps; no `wall_touch` before navigation. |
| `test_repeated_colours_each_get_a_cue_event` | With consecutive duplicates allowed, every cue item (including repeats) gets its own event and the cue events reproduce the sequence. |
| `test_wrong_wall_correct_wall_and_completion` | A wrong wall gives −1, the same room and a re-centred agent; correct walls give `wall_touch` + `room_reset`; the last one gives +10 and `success`. |
| `test_navigation_info_never_reveals_the_target` | Without `debug_info`, `info` only ever contains `events` (plus the end-of-episode fields), and navigation events are limited to `nav_start`, `wall_touch` and `room_reset`. |
| `test_truncation_at_episode_limit` | Reaching `max_steps` truncates with cause `timeout`. |
| `test_seeded_episodes_are_deterministic_and_noise_uses_its_own_rng` | The same seed gives identical frames; adding `GaussianNoiseWrapper` does not change the sequence. |
| `test_rgb_colours_are_distinct_for_the_vocabulary` | The 6 vocabulary colours are distinct RGB values. |
| `test_cue_rm_requires_symmetric_rooms` | The cue-recording RM refuses fixed wall colours. |
| `test_oracle_solves_final_eval_seeds` (slow) | The oracle succeeds on all 200 final-evaluation seeds within 150 steps. |

### 4. Environment Factory, Seeding and SB3 (`test_factory_seeding_sb3.py`)
> 📁 [`tests/test_factory_seeding_sb3.py`](../tests/test_factory_seeding_sb3.py)

| Test | Checks |
|---|---|
| `test_code_is_imported_from_this_checkout` | `temporal_comp_gym` and `agents` are imported from this repository, not from another install. |
| `test_frozen_main_conditions` | Decisions D1 (delivered count hidden) and D2 (distractors on), step limits 500/300/4000, observation shapes (8,), (84, 84, 3), (28,), and that unknown overrides are rejected. |
| `test_seed_sets_are_disjoint` | Training, training-env and evaluation seed sets never overlap. |
| `test_run_streams_are_reproducible_and_independent` | Per-run random streams are reproducible and distinct from each other. |
| `test_sb3_receives_truncation_flags_from_internal_limits` | SB3's vector env sees the internal time limit as a truncation (`TimeLimit.truncated`). |
| `test_ppo_rm_and_task_state_controls_build_and_train` | PPO-RM (28 + 8 inputs), DQN-TS delivery (8 + 7) and DQN-TS colour (image + task state) build and train briefly. |

### 5. Reward Machines (`test_reward_machines.py`)
> 📁 [`tests/test_reward_machines.py`](../tests/test_reward_machines.py)

| Test | Checks |
|---|---|
| `test_rm_reward_terminal_and_state_match_the_env` (×6: 3 tasks × oracle/random policy) | On every step the RM reward and terminal flag equal the environment's, and the RM state equals the hidden task state. |
| `test_rm_memory_goes_beyond_the_observation` | The delivery RM tracks (delivered, carried) in 7 states; the maintenance features encode held codes, stage and readiness. |
| `test_colour_register_is_built_from_events_alone` | Fed only the recorded event stream, the cue-recording RM reconstructs the exact sequence. |
| `test_old_wrapper_module_is_retired` | Importing `agents.rm_wrapper` raises `ImportError`. |
| `test_delivery_counterfactual_validity_matches_env_dynamics` | For every step, a cloned environment with each other carried count produces the same event exactly when the RM marks that state valid. |
| `test_colour_counterfactual_validity_matches_env_dynamics` | For every navigation head, a cloned environment with that target and walls remaining produces the same frame, reward and termination exactly when the RM marks the head valid. |
| `test_counterfactual_rules_reproduce_the_visited_transition` | The counterfactual tables always include the visited head, with its real reward, next head and terminal flag. |

### 6. QRM Learner (`test_qrm.py`)
> 📁 [`tests/test_qrm.py`](../tests/test_qrm.py)

| Test | Checks |
|---|---|
| `test_vectorised_targets_match_a_per_head_loop` (delivery, colour) | The vectorised QRM targets equal a slow per-head computation. |
| `test_truncation_bootstraps_but_rm_terminal_does_not` | Truncated transitions bootstrap; RM-terminal ones do not. |
| `test_replay_stores_rm_rewards_heads_and_event_classes` | The replay buffer stores valid rewards, heads and event classes. |
| `test_qrm_updates_use_counterfactual_heads` (delivery, colour) | The update count matches the schedule, losses are finite, and on average more than one head is updated per transition. |

### 7. Replay Buffer and DRQN (`test_replay_and_drqn.py`)
> 📁 [`tests/test_replay_and_drqn.py`](../tests/test_replay_and_drqn.py)

| Test | Checks |
|---|---|
| `test_transitions_survive_wraparound` | Transitions stay correct when the circular buffer wraps around. |
| `test_sequences_never_cross_episodes_and_start_from_stored_state` (two settings) | Sampled sequences never cross episodes, start from the stored recurrent state, mask burn-in and padding, and (with alignment on) mostly start at episode starts. |
| `test_image_replay_memory_is_bounded` | The colour image replay uses about one `uint8` frame (plus recurrent state) per slot, under 2.5 GB at the full size. |
| `test_hidden_state_advances_on_random_actions` | The recurrent state moves on even for random actions, and an episode starts from a zero state. |
| `test_truncation_bootstraps_and_termination_does_not` | DRQN targets bootstrap on truncation only. |
| `test_drqn_learns_a_delayed_recall_task` (slow) | DRQN learns a small cue-then-delay recall task to at least 90% success, which a memoryless policy cannot exceed 50% on. |
| `test_drqn_colour_smoke_runs_updates_on_rgb_frames` | DRQN on Sequential Colour stores RGB `uint8` frames and runs finite updates on schedule. |

### 8. Baseline Pipeline (`test_pipeline.py`)
> 📁 [`tests/test_pipeline.py`](../tests/test_pipeline.py)

These tests run the real entry-point scripts as subprocesses on tiny smoke budgets.

| Test | Checks |
|---|---|
| `test_plan_covers_the_agreed_matrix` | 55 runs (10 controls), 25/15/15 per machine role, seed-major order with controls last, equal budgets within an environment, and `baseline_plan.json` is up to date. |
| `test_config_hash_overrides_and_guards` | Config hashes change with seed, mode and overrides but not with the W&B entity; the default project is `new_experiments`; unknown overrides, protected keys, the pilot W&B project, out-of-range seeds and budgets SB3 would round are rejected. |
| `test_smoke_run_writes_complete_isolated_records` | A smoke run writes a complete, verified run directory (status, identity, config, metadata, CSVs, offline W&B, restart checkpoint) with no API key in any file. |
| `test_fresh_runs_never_collide_or_inherit_state` | Repeated launches get separate directories and IDs, and start from an identical untrained policy. |
| `test_injected_failure_is_recorded_as_failed_not_zero` | An injected failure exits 1, is recorded as `failed` with no summary, and the manifest reports it as failed. |
| `test_resume_restores_the_run` (slow; DRQN and DQN) | After an injected failure, `--resume` refuses the wrong seed, completes the run with the same update count as an uninterrupted run and no duplicate evaluations, and refuses a second resume. |
| `test_evaluation_does_not_touch_training_state` | Evaluation changes no training random state, replay or weights, is repeatable for the same seeds, and resets the recurrent state. |
| `test_launcher_requires_an_explicit_interpreter_and_lists_the_matrix` | A launcher without `--python` refuses to start; `--dry-run` lists the 25 memoryless runs with the right device, task-state and W&B flags. |
| `test_launcher_reports_failures_and_exits_nonzero` | With failing fake runs the launcher prints `BATCH FAILED`, records exit codes and per-run logs, and exits 1; with succeeding ones it exits 0. |
| `test_old_remediation_script_is_deprecated` | The old remediation script refuses to run. |

### 9. Batch Restart (`test_continue_batch.py`)
> 📁 [`tests/test_continue_batch.py`](../tests/test_continue_batch.py)

| Test | Checks |
|---|---|
| `test_batch_action_fresh_resume_skip` (slow) | `--continue-batch` decides correctly: start fresh when no run exists, resume an unfinished run (ignoring runs with a different config hash), skip a complete one; the launcher's dry run shows the skip. |

---

## Related Verification Scripts (not part of pytest)

These live in `run_scripts/baseline/` and write JSON reports to `run_scripts/baseline/validation/`:

*   **`validate_tasks.py`**: Runs the scripted oracle and a random policy on the 200 final-evaluation seeds of each task and checks every step's reward-machine reward. The recorded result is 100% oracle success and 0% random success on all three tasks.
*   **`validate_pipeline.py`**: An offline end-to-end check of the whole training pipeline: one smoke run per environment/algorithm combination, resume, collisions, failure cases, configuration rejection, environment checks, replay memory and the manifest. `--pytest` also runs this test suite.

```bash
D:\miniconda3\envs\temporal_gym\python.exe run_scripts/baseline/validate_tasks.py
D:\miniconda3\envs\temporal_gym\python.exe run_scripts/baseline/validate_pipeline.py
```
