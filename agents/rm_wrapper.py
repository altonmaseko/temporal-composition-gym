"""Retired in protocol v2.

The old proposition wrappers inferred events from reward values (lava was labelled as a
premature goal), emitted `correct_wall` during the colour cue phase and tracked only task
state that was already observable. Use agents.reward_machines instead, which reads the
environments' explicit info["events"] stream.

Importing this module fails loudly so that legacy tuning/stress scripts cannot silently
run with the old semantics.
"""
raise ImportError(
    "agents/rm_wrapper.py was retired in protocol v2; use agents.reward_machines "
    "(see run_scripts/baseline/BASELINE_PROTOCOL.md section 4)"
)
