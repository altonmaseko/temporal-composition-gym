# Task and agent changes for protocol v2 (Prompt 2)

What changed relative to the pilot code (`main` at `972192a`), why, and how it is tested.
The three tasks keep their structure: deliver packages then exit; memorise a colour
sequence then touch walls in order; collect maintenance codes, service them once ready,
then exit the last stage.

## Hazardous Delivery

| Change | Why | Test |
|---|---|---|
| Explicit `info["events"]` and `termination_cause` / `is_success` | Lava was labelled "premature goal" by comparing −50 rewards | `test_lava_is_not_premature_goal`, `test_damage_fatal_damage_and_healing` |
| Main condition hides the delivered count (D1) | Otherwise the only task memory is one carried bit | `test_main_condition_hides_delivered_count` |
| Layouts where lava walls off an objective are redrawn (seeded) | 0.4% of pilot layouts were unsolvable | `test_every_generated_layout_is_solvable` (1,000 seeds) |
| Truncation at `max_steps=500` only when not terminated; trainers' extra `TimeLimit(1000)` removed | The 1,000 limit never applied; cause must be recorded | `test_truncation_is_not_termination` |
| `task_dependencies=True` raises | Not validated for the RM / counterfactual rules | `test_unsupported_task_dependencies_raise` |
| Fixed start cell reserved before objects are placed | Objects could be placed on the start cell | — |

Unchanged on purpose: packages and destinations share sensor value 3; walls read 0; a
cell's effect applies again when the agent bumps into a wall and stays on it.

## Sequential Colour

| Change | Why | Test |
|---|---|---|
| Events: `cue(colour)` per cue item (first one in reset info), `delay_start`, `nav_start`, `wall_touch(colour)`, `room_reset` | The pilot RM saw "correct_wall" during the cue phase | `test_cue_delay_and_nav_event_timing`, `test_repeated_colours_each_get_a_cue_event` |
| Internal limit `max_steps=300` with `timeout` cause | No limit existed in the env | `test_truncation_at_episode_limit` |
| Navigation `info` never contains the target | No hidden-answer leakage | `test_navigation_info_never_reveals_the_target` |
| `frames_per_colour` is a constructor argument (default 5, as before) | Was a hard-coded attribute | — |
| Noise wrapper uses its own seeded generator | It used global NumPy randomness | `test_seeded_episodes_are_deterministic_and_noise_uses_its_own_rng` |

## Deferred Maintenance

| Change | Why | Test |
|---|---|---|
| All randomness from the seeded env RNG; a fresh physics space per episode | Same seed gave different layouts; `check_env` failed | `test_gymnasium_env_checker_passes`, `test_seeded_layout_and_rollout_are_deterministic` |
| Seeded episode schedule: readiness stage r ∈ [1, 2]; one distinct code type in each stage before r; codes only there | 58% of pilot episodes could not be completed (duplicate deadlock, no matching station) | `test_schedule_is_solvable_by_construction` |
| Every stage has one station per type; with `distractor_stations=True` (D2) that is all 3 types | Station layout must not reveal held codes; without distractors every station is correct | same |
| `readiness_delay_range` is inclusive and validated (`1 ≤ min ≤ max ≤ max_stages−1`); default `(1, 2)` gives the same {1, 2} as the old exclusive `(1, 3)` | Ranges ≥ the stage count could never activate | `test_invalid_or_unimplemented_configurations_raise` |
| `score` is the plain running sum of step rewards | It added already-accumulated rewards a second time | `test_score_is_the_running_sum_of_rewards` |
| Macro waypoint implemented: offset to the next path tile towards the exit | It was always `[0, 0]` | `test_macro_waypoint_points_to_next_path_tile` |
| Events and causes: `code_collected`, `service_attempt`, `readiness_on`, `stage_exit`, `final_exit`, `stage_exit_incomplete`, `boundary_contact`, `off_path` | The pilot RM parsed score differences | `test_code_collection_event_and_tile_conversion`, `test_premature_success_and_wrong_service`, `test_exit_outcomes`, `test_readiness_switches_on_when_entering_the_readiness_stage` |
| Internal limit `max_steps=4000` (≥ 2× oracle p95 of 1,859 steps) | No limit existed in the env; the protocol's 3,000 was provisional | `test_truncation_at_episode_limit`, `test_oracle_success_and_horizon` |
| Interactables placed ≥ 2 cells apart and from the exit; the walk is lengthened when they do not fit | Hovering on one must not touch another or the exit | oracle tests |
| Local-grid sensor and off-path check use the stage's cell map; the radar reports the nearest station of each type | Point queries could return the rocket's own shape; the radar returned the first station found | — |
| `duration_matching_mode=True` raises; hand-authored `CUSTOM_STAGES` removed | Advertised toggle had no effect; fixed layouts bypass the solvable schedule | — |
| Baselines see `MaintenanceScaledObservation` (28 values; radar as present/dx/dy) | Raw pixel units and the −99 sentinel; fixed scaling needs no saved statistics | `test_frozen_main_conditions` |

Parameter name: the colour and maintenance step limit is `max_steps` (as in delivery), not
`max_episode_steps`, because `gym.make` consumes `max_episode_steps` and adds its own
`TimeLimit`, which would bypass the env's limit and its `timeout` cause.

## Reward machines (`agents/reward_machines/`)

Driven only by `info["events"]` and fixed configuration. `RewardMachineWrapper` raises if the
RM reward or terminal flag ever differs from the env's. `agents/rm_wrapper.py` is retired.

| Machine | State (task memory) | Tests |
|---|---|---|
| Delivery | (delivered, carried): 7 live states | RM state equals the hidden counters on oracle and random rollouts; counterfactual validity checked against cloned envs with every carried count |
| Maintenance | held codes, stage, ready → 8 features for PPO-RM | RM state equals hidden held codes and stage |
| Colour (QRM-Cue) | recorded colour sequence + walls done; 40 Q heads | register built from events alone equals the sequence; counterfactual validity checked against cloned envs for all 30 navigation heads, in every case (no touch, wrong, correct, completion) |

## Learners

- **DRQN** (`agents/drqn/drqn_core.py`): NatureCNN on RGB for colour; frame replay sized in
  frames (uint8, one copy, ~2.3 GB at 100k frames incl. stored LSTM state); state advances on
  ε-random steps; sequences never cross episodes, start from the stored actor state with
  burn-in, padding masked; truncation bootstraps. Tests: `tests/test_replay_and_drqn.py`,
  including learning a delayed-recall task to ≥ 90%.
- **QRM / QRM-Cue** (`agents/qrm/qrm_core.py`): vectorised counterfactual targets over all valid
  heads; RM terminal ends bootstrapping, truncation does not. Tests: `tests/test_qrm.py`.
- **SB3 trainers** build environments through `agents/common/env_factory.py`. PPO-RM uses the new
  maintenance RM and the untuned common PPO defaults. Their checkpoint/W&B/evaluation code is
  still the pilot version until Prompt 3.

## Refinements to the frozen protocol

Recorded in `BASELINE_PROTOCOL.md` §11: maintenance limit 4,000; inclusive readiness range;
`max_steps` naming; maintenance scaled observation has 28 values; DRQN colour aligns half its
sequences to the episode start; gradients flow through burn-in frames.
