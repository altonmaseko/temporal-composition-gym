"""Stress-design probe: proposed machine assignment and expected hours (no runs, no seeds used).

Per-run hours = max(tuning-hardware projection from out/runtime_model.json,
                    v2 RTX 3090 full-run time + the noisy half of the projected evaluation),
plus +25% evaluation time for maintenance levels whose step limit rises from 4,000 to 5,000.
A "set" is all 11 cells at one (level, training seed). Output: out/machine_plan.json
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "analysis", "stress_design", "out")

# Order inside a set: cheap CPU cells first, colour DRQN (longest) last.
CELL_ORDER = ["delivery/dqn", "delivery/dqn_ts", "delivery/drqn", "delivery/qrm",
              "maintenance/ppo", "maintenance/recurrent_ppo", "maintenance/ppo_rm",
              "colour/dqn", "colour/dqn_ts", "colour/qrm_cue", "colour/drqn"]
RAISED_LIMIT_LEVELS = {"hard1", "hard2"}        # maintenance only

# Rounds: each machine's k-th set. Round 1 = three RQ1 seeds; round 2 = one seed of every
# stress level; round 3 completes RQ1; rounds 4-5 complete the stress levels.
ASSIGN = {
    "a": [("default", 301), ("easy", 201), ("default", 304), ("hard1", 202), ("hard1", 203)],
    "b": [("default", 302), ("hard1", 201), ("default", 305), ("hard2", 202), ("hard2", 203)],
    "c": [("default", 303), ("hard2", 201), ("easy", 202), ("easy", 203)],
}
# Balance: the colour DRQN runs of A's and B's last sets move to the end of C's queue.
MOVED = [("a", ("hard1", 203), "colour/drqn"), ("b", ("hard2", 203), "colour/drqn")]


def per_run_hours():
    rm = json.load(open(os.path.join(OUT, "runtime_model.json"), encoding="utf-8"))["cells"]
    h = {}
    for cell, v in rm.items():
        proj = v["projected_full_total_h"]
        v2 = v["v2_full_h_mean"]
        alt = v2 + v["projected_full_eval_h"] / 2 if v2 is not None else proj
        h[cell] = {"hours": max(proj, alt), "eval_h": v["projected_full_eval_h"],
                   "tuning_hw_projection_h": proj, "v2_plus_noisy_h": alt}
    return h


def main():
    hrs = per_run_hours()
    queues = {r: [] for r in ASSIGN}
    for role, sets in ASSIGN.items():
        for level, seed in sets:
            for cell in CELL_ORDER:
                if (role, (level, seed), cell) in MOVED:
                    continue
                queues[role].append((level, seed, cell))
    for role, (level, seed), cell in MOVED:
        queues["c"].append((level, seed, cell))

    def run_h(level, cell):
        x = hrs[cell]["hours"]
        if cell.startswith("maintenance") and level in RAISED_LIMIT_LEVELS:
            x += 0.25 * hrs[cell]["eval_h"]
        return x

    out = {"per_run_hours": hrs, "machines": {}}
    total = 0.0
    for role, q in queues.items():
        t, rows = 0.0, []
        for i, (level, seed, cell) in enumerate(q):
            t += run_h(level, cell)
            rows.append({"slot": i, "level": level, "seed": seed, "cell": cell, "cum_h": round(t, 2)})
        colour = sum(run_h(lv, c) for lv, s, c in q if c.startswith("colour"))
        out["machines"][role] = {"runs": len(q), "hours": t, "colour_hours": colour, "queue": rows}
        total += t
        print(f"machine {role}: {len(q)} runs, {t:.1f} h (colour {colour:.1f} h)")
    out["total_hours"] = total
    print(f"total {sum(len(q) for q in queues.values())} runs, {total:.1f} machine-hours")
    set_h = sum(hrs[c]["hours"] for c in CELL_ORDER)
    print(f"one default-level set: {set_h:.2f} h; colour share {sum(hrs[c]['hours'] for c in CELL_ORDER if c.startswith('colour'))/set_h:.0%}")
    with open(os.path.join(OUT, "machine_plan.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
