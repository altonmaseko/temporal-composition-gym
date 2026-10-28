"""Step 4: targeted checks of suspected failure modes and pipeline bugs (read-only, no learning).

Sections (each writes into out/suspects.json):
  A  discount horizon: discounted return (gamma 0.99) of the oracle, of random play and of the
     learned "stall" behaviours, plus when the oracle's rewards arrive
  B  reward density in replay: rewarding transitions inside the 100k replay window over
     training, and the chance a minibatch contains one
  C  Q-value bounds: Q over every head on visited observations vs the largest achievable return
  D  PPO action saturation (maintenance): pre-clip action means, policy std, value predictions
  E  synthetic FrameReplay check: transitions and R2D2 sequences never cross episodes, next
     frames and stored LSTM states line up, padding is masked (incl. ring-buffer wrap-around)
  F  plumbing check: rewards / terminated / truncated reaching each learner's buffer equal the
     env's (SB3 DQN, SB3 PPO, DRQN, QRM) -- data collection only, zero gradient updates
Diagnostic seeds only: training seed 401, env/episode seeds 6,000,000+ (except A/C/D, which
reuse the first periodic-evaluation episode seeds, as for the behaviour rollouts).

    python -B analysis/baseline_v2/04_suspects.py
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from common import CELLS, DIAG_EPISODE_SEED_BASE, OUT, SEEDS, read_json, run_dirs, write_json
from agents.common.env_factory import make_env
from agents.common.runner import build_env
from agents.common.seeding import EVAL_EPISODE_SEED_BASE, RunRNG
from temporal_comp_gym.oracles import ORACLES

torch.set_num_threads(2)
GAMMA = 0.99
RUNS = run_dirs()
PERIODIC = EVAL_EPISODE_SEED_BASE["periodic"]
res = {}
only = set(sys.argv[1:])


def want(s):
    return not only or s in only


def disc(rewards, g=GAMMA):
    return float(sum((g ** t) * r for t, r in enumerate(rewards)))


# ============================================================ A. discount horizon
if want("A"):
    A = {}
    for env_key in CELLS:
        n = 5 if env_key == "maintenance" else 20
        env = make_env(env_key, debug_info=True)
        rows = []
        for i in range(n):
            orc = ORACLES[env_key](env)
            env.reset(seed=PERIODIC + i)
            orc.reset()
            rs = []
            while True:
                _, r, term, trunc, info = env.step(orc.act())
                rs.append(r)
                if term or trunc:
                    break
            pos_t = [t for t, r in enumerate(rs) if r > 0]
            rows.append({"return": sum(rs), "disc_0.99": disc(rs), "disc_0.999": disc(rs, 0.999), "length": len(rs),
                         "success": bool(info.get("is_success")), "first_pos_reward_step": pos_t[0] if pos_t else None,
                         "last_pos_reward_step": pos_t[-1] if pos_t else None,
                         "discount_weight_of_final_reward": GAMMA ** (len(rs) - 1)})
        # uniform random reference (same seeds)
        rnd = []
        rng = np.random.default_rng(DIAG_EPISODE_SEED_BASE)
        for i in range(n):
            env.reset(seed=PERIODIC + i)
            rs = []
            while True:
                a = rng.uniform(-1, 1, 2) if env_key == "maintenance" else int(rng.integers(4))
                _, r, term, trunc, info = env.step(a)
                rs.append(r)
                if term or trunc:
                    break
            rnd.append({"return": sum(rs), "disc_0.99": disc(rs), "length": len(rs), "cause": info.get("termination_cause")})
        df, dr = pd.DataFrame(rows), pd.DataFrame(rnd)
        A[env_key] = {"oracle": {k: (float(df[k].mean()) if df[k].notna().any() else None) for k in df.columns if k != "success"},
                      "oracle_success": float(df["success"].mean()),
                      "random": {"return": float(dr["return"].mean()), "disc_0.99": float(dr["disc_0.99"].mean()),
                                 "length": float(dr["length"].mean()), "causes": dr["cause"].value_counts().to_dict()},
                      "episodes": n, "effective_horizon_steps_1_over_1_minus_gamma": 1 / (1 - GAMMA)}
        env.close()
        print("A", env_key, json.dumps(A[env_key])[:400], flush=True)
    # analytic stall values for maintenance: lying on the boundary vs hovering off-path, gamma 0.99
    A["maintenance"]["stall_values_gamma_0.99"] = {
        "boundary_every_step_(-5.1/step)": -5.1 / (1 - GAMMA),
        "off_path_every_step_(-0.1/step)": -0.1 / (1 - GAMMA),
        "exit_with_uncollected_code_after_200_steps": -1000 * GAMMA ** 200,
        "one_service_after_600_steps": 100 * GAMMA ** 600,
        "perfect_completion_after_1400_steps": 1000 * GAMMA ** 1400,
    }
    res["A_discount_horizon"] = A

# ============================================================ B. reward density in replay
if want("B"):
    B = {}
    for env_key, algos in CELLS.items():
        if env_key == "maintenance":
            continue
        for algo in algos:
            per_seed = []
            for s in SEEDS:
                ep = pd.read_csv(os.path.join(RUNS[(env_key, algo, s)], "train_episodes.csv"))
                budget = 1_000_000 if env_key == "delivery" else 3_000_000
                bins = np.arange(0, budget + 1, 100_000)
                # rewarding transitions: success terminals; for delivery also each +10 delivery
                succ = np.histogram(ep.loc[ep["success"] == 1, "global_step"], bins)[0]
                rew = succ.copy()
                if env_key == "delivery":
                    rew = np.histogram(np.repeat(ep["global_step"].values, ep["progress"].values.astype(int)), bins)[0] + succ
                per_seed.append({"success_per_100k": succ.tolist(), "rewarding_per_100k": rew.tolist()})
            succ = np.mean([p["success_per_100k"] for p in per_seed], axis=0)
            rew = np.mean([p["rewarding_per_100k"] for p in per_seed], axis=0)
            late_s, late_r = float(succ[-5:].mean()), float(rew[-5:].mean())
            B[f"{env_key}/{algo}"] = {
                "success_transitions_in_replay_window_mean_by_100k": succ.round(2).tolist(),
                "rewarding_transitions_in_replay_window_mean_by_100k": rew.round(2).tolist(),
                "late_success_transitions_per_100k": late_s,
                "late_P(minibatch_of_32_has_a_success)": 1 - (1 - late_s / 100_000) ** 32,
                "late_rewarding_transitions_per_100k": late_r,
                "late_P(minibatch_of_32_has_any_positive_reward)": 1 - (1 - late_r / 100_000) ** 32,
            }
            print("B", env_key, algo, B[f"{env_key}/{algo}"]["late_success_transitions_per_100k"],
                  B[f"{env_key}/{algo}"]["late_rewarding_transitions_per_100k"], flush=True)
    res["B_replay_reward_density"] = B

# ============================================================ C. Q-value bounds
if want("C"):
    from stable_baselines3 import DQN
    from agents.drqn.drqn_core import DRQNAgent, DRQNConfig
    from agents.qrm.qrm_core import QRMAgent, QRMConfig
    bounds = {"delivery": 80.0, "colour": 10.0}        # largest undiscounted return of any episode
    C = {}
    for env_key in ("delivery", "colour"):
        for algo in CELLS[env_key]:
            stats = []
            for s in SEEDS:
                d = RUNS[(env_key, algo, s)]
                cfg = read_json(os.path.join(d, "config.resolved.json"))
                env = build_env(cfg)
                step = cfg["budget"]["env_steps"]
                ext = "zip" if algo in ("dqn", "dqn_ts") else "pt"
                path = os.path.join(d, "checkpoints", f"policy_step{step:09d}.{ext}")
                rng = np.random.default_rng([DIAG_EPISODE_SEED_BASE, s, 7])
                qs_all, qs_taken = [], []
                if algo in ("dqn", "dqn_ts"):
                    m = DQN.load(path, device="cpu")
                    qf = lambda o: m.q_net(m.policy.obs_to_tensor(o)[0])[0].detach().numpy()[None]  # noqa: E731
                elif algo == "drqn":
                    c = dataclasses.replace(DRQNConfig(**cfg["algo"]["hyperparams"]), replay_frames=16)
                    ag = DRQNAgent(env.observation_space, env.action_space.n, c, "cpu", RunRNG(401))
                    ag.q.load_state_dict(torch.load(path, map_location="cpu"))
                else:
                    c = dataclasses.replace(QRMConfig(**cfg["algo"]["hyperparams"]), replay_frames=16)
                    ag = QRMAgent(env.observation_space, env.action_space.n, env.env.rm, c, "cpu", RunRNG(401))
                    ag.q.load_state_dict(torch.load(path, map_location="cpu"))
                for i in range(3):
                    obs, info = env.reset(seed=PERIODIC + i)
                    h = ag.zero_hidden() if algo == "drqn" else None
                    head = info["rm"]["head"] if algo in ("qrm", "qrm_cue") else None
                    while True:
                        with torch.no_grad():
                            if algo == "drqn":
                                q, h = ag.policy_step(obs, h)
                                allq, q_act = q[None], q
                            elif algo in ("qrm", "qrm_cue"):
                                allq = ag.q(torch.as_tensor(np.asarray(obs)[None]))[0].numpy()   # (H, A)
                                q_act = allq[head]
                            else:
                                allq = qf(obs)
                                q_act = allq[0]
                        qs_all.append(allq.max(-1))
                        qs_taken.append(q_act.max())
                        a = int(np.argmax(q_act)) if rng.random() >= 0.05 else int(rng.integers(4))
                        obs, r, term, trunc, info = env.step(a)
                        if term or trunc:
                            break
                        if head is not None:
                            head = info["rm"]["next_head"]
                env.close()
                qa = np.concatenate([np.ravel(x) for x in qs_all])
                qt = np.array(qs_taken)
                stats.append({"seed": s, "maxQ_current_head_mean": float(qt.mean()), "maxQ_current_head_max": float(qt.max()),
                              "maxQ_all_heads_mean": float(qa.mean()), "maxQ_all_heads_max": float(qa.max()),
                              "frac_all_heads_above_bound": float((qa > bounds[env_key]).mean()),
                              "frac_current_head_above_bound": float((qt > bounds[env_key]).mean())})
            C[f"{env_key}/{algo}"] = {"bound_undiscounted_max_return": bounds[env_key], "per_seed": stats}
            print("C", env_key, algo, [(x["maxQ_current_head_mean"], x["maxQ_all_heads_max"]) for x in stats], flush=True)
    res["C_q_bounds"] = C

# ============================================================ C2. QRM-Cue Q by head group on the frames each group is trained on
if want("C2"):
    from agents.qrm.qrm_core import QRMAgent, QRMConfig
    C2 = {}
    for s in SEEDS:
        d = RUNS[("colour", "qrm_cue", s)]
        cfg = read_json(os.path.join(d, "config.resolved.json"))
        env = build_env(cfg)
        rm = env.env.rm
        c = dataclasses.replace(QRMConfig(**cfg["algo"]["hyperparams"]), replay_frames=16)
        ag = QRMAgent(env.observation_space, env.action_space.n, rm, c, "cpu", RunRNG(401))
        per_snap = {}
        for step in (300_000, 1_500_000, 3_000_000):
            ag.q.load_state_dict(torch.load(os.path.join(d, "checkpoints", f"policy_step{step:09d}.pt"), map_location="cpu"))
            rng = np.random.default_rng([DIAG_EPISODE_SEED_BASE, s, 8])
            nav_q, pre_q, cur_q, present_q, absent_q = [], [], [], [], []
            for i in range(3):
                obs, info = env.reset(seed=PERIODIC + i)
                head = info["rm"]["head"]
                while True:
                    with torch.no_grad():
                        allq = ag.q(torch.as_tensor(np.asarray(obs)[None]))[0].numpy().max(-1)     # (H,)
                    core = env.unwrapped
                    if core.current_phase == "navigation":
                        nav_q.append(allq[rm.is_nav])
                        walls = {core.colour_id(v) for v in core.wall_colors.values()}
                        pres = np.array([c_ in walls for c_ in rm.head_c[rm.is_nav]])
                        present_q.append(allq[rm.is_nav][pres])
                        absent_q.append(allq[rm.is_nav][~pres])
                    else:
                        pre_q.append(allq[~rm.is_nav])
                    cur_q.append(allq[head])
                    a = int(np.argmax(ag.q_values(obs, head))) if rng.random() >= 0.05 else int(rng.integers(4))
                    obs, r, term, trunc, info = env.step(a)
                    if term or trunc:
                        break
                    head = info["rm"]["next_head"]
            cat = lambda xs: np.concatenate([np.ravel(x) for x in xs]) if xs else np.array([np.nan])  # noqa: E731
            nq, pq, cq, prq, abq = cat(nav_q), cat(pre_q), np.array(cur_q), cat(present_q), cat(absent_q)
            per_snap[step] = {"nav_heads_on_nav_frames": {"mean": float(nq.mean()), "max": float(nq.max()),
                                                          "frac_above_10": float((nq > 10).mean())},
                              "nav_heads_target_on_a_wall": {"mean": float(prq.mean()), "max": float(prq.max())},
                              "nav_heads_target_not_on_any_wall": {"mean": float(abq.mean()), "max": float(abq.max()),
                                                                   "frac_above_10": float((abq > 10).mean())},
                              "pre_heads_on_pre_frames_max": float(np.nanmax(pq)),
                              "current_head": {"mean": float(cq.mean()), "max": float(cq.max())}}
        env.close()
        C2[s] = per_snap
        print("C2 seed", s, {k: (round(v["nav_heads_on_nav_frames"]["max"], 1), round(v["nav_heads_target_not_on_any_wall"]["max"], 1),
                                 round(v["nav_heads_target_on_a_wall"]["max"], 1)) for k, v in per_snap.items()}, flush=True)
    res["C2_qrm_cue_heads"] = C2

# ============================================================ D. PPO saturation (maintenance)
if want("D"):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    D = {}
    for algo in CELLS["maintenance"]:
        stats = []
        for s in SEEDS:
            d = RUNS[("maintenance", algo, s)]
            cfg = read_json(os.path.join(d, "config.resolved.json"))
            env = build_env(cfg)
            path = os.path.join(d, "checkpoints", "policy_step002048000.zip")
            m = (RecurrentPPO if algo == "recurrent_ppo" else PPO).load(path, device="cpu")
            obs, _ = env.reset(seed=PERIODIC)
            means, values, rews = [], [], []
            state, start = None, np.ones((1,), dtype=bool)
            for t in range(1500):
                with torch.no_grad():
                    ot = m.policy.obs_to_tensor(obs)[0]
                    if algo == "recurrent_ppo":
                        a, state = m.predict(obs, state=state, episode_start=start, deterministic=True)
                        start[:] = False
                        mean = np.array(a, dtype=float).reshape(-1)        # clipped; raw mean not exposed cheaply
                        v = None
                    else:
                        dist = m.policy.get_distribution(ot)
                        mean = dist.distribution.mean[0].numpy()
                        v = float(m.policy.predict_values(ot)[0, 0])
                        a = np.clip(mean, -1, 1)
                means.append(mean)
                values.append(v)
                obs, r, term, trunc, info = env.step(a)
                rews.append(r)
                if term or trunc:
                    break
            env.close()
            means = np.array(means)
            log_std = m.policy.log_std.detach().numpy().tolist() if hasattr(m.policy, "log_std") else None
            vals = [v for v in values if v is not None]
            stats.append({"seed": s, "mean_action_mean": means.mean(0).tolist(),
                          "frac_thrust_mean_below_-1": float((means[:, 0] < -1).mean()),
                          "frac_thrust_mean_above_0": float((means[:, 0] > 0).mean()),
                          "frac_torque_mean_saturated": float((np.abs(means[:, 1]) >= 1).mean()),
                          "policy_std": None if log_std is None else np.exp(log_std).tolist(),
                          "value_at_reset": vals[0] if vals else None,
                          "disc_return_from_reset_first_1500_steps": disc(rews),
                          "steps": len(rews)})
        D[algo] = stats
        print("D", algo, [(round(x["mean_action_mean"][0], 2), round(x["mean_action_mean"][1], 2), x["value_at_reset"],
                           round(x["disc_return_from_reset_first_1500_steps"], 1)) for x in stats], flush=True)
    res["D_ppo_saturation"] = D

# ============================================================ E. synthetic FrameReplay check
if want("E"):
    from agents.common.replay import FrameReplay
    rng = np.random.default_rng(DIAG_EPISODE_SEED_BASE + 1)
    cap = 97
    rp = FrameReplay(cap, (1,), np.float32, hidden_shape=(2, 3))
    truth = {}                      # (episode, t) -> frame value; frame value encodes both
    ep_id = 0
    errors = []
    for e in range(60):
        L = int(rng.integers(1, 25))
        trunc_end = bool(rng.random() < 0.3)
        def val(t):
            return np.array([e * 1000 + t], dtype=np.float32)
        rp.start_episode(val(0), np.full((2, 3), e * 1000 + 0, dtype=np.float32))
        for t in range(L):
            last = t == L - 1
            rp.add(action=t % 4, reward=float(e * 1000 + t), terminated=last and not trunc_end, truncated=last and trunc_end,
                   next_obs=val(t + 1), next_hidden=np.full((2, 3), e * 1000 + t + 1, dtype=np.float32))
    n_checked = 0
    for _ in range(300):
        b = rp.sample_transitions(16, rng)
        for o, no, r in zip(b["obs"][:, 0], b["next_obs"][:, 0], b["reward"]):
            n_checked += 1
            if not (no == o + 1 and r == o):
                errors.append(("transition", float(o), float(no), float(r)))
        for frac in (0.0, 0.5, 1.0):
            sq = rp.sample_sequences(4, 8, 4, rng, episode_start_fraction=frac)
            for i in range(4):
                o = sq["obs"][i, :, 0]
                m = sq["loss_mask"][i]
                k = int(sq["burn_in"][i])
                epi = int(o[0] // 1000)
                if sq["h0"][i, 0, 0] != o[0]:
                    errors.append(("h0", float(sq["h0"][i, 0, 0]), float(o[0])))
                if frac == 1.0 and o[0] % 1000 != 0 and k == 0:
                    pass                          # whole prefix may have been overwritten: allowed
                for pos in range(o.shape[0] - 1):
                    if m[pos] == 1:
                        n_checked += 1
                        if not (o[pos + 1] == o[pos] + 1 and int(o[pos] // 1000) == epi and sq["reward"][i, pos] == o[pos]):
                            errors.append(("seq", i, pos, o.tolist(), m.tolist()))
                        if pos < k:
                            errors.append(("mask_in_burn_in", i, pos))
                    if sq["terminated"][i, pos] or sq["truncated"][i, pos]:
                        if m[pos + 1:].any():
                            errors.append(("mask_after_end", i, pos))
    res["E_replay_synthetic"] = {"capacity": cap, "episodes_written": 60, "checks": n_checked, "errors": errors[:10],
                                 "n_errors": len(errors)}
    print("E", res["E_replay_synthetic"]["checks"], "errors", len(errors), flush=True)

# ============================================================ F. plumbing (no learning)
if want("F"):
    from agents.common import baseline_plan as plan
    F = {}
    seed = 401
    env_seed = DIAG_EPISODE_SEED_BASE + 401

    def sb3_dqn_check(env_key, algo, steps=3000):
        from stable_baselines3 import DQN
        from stable_baselines3.common.vec_env import DummyVecEnv
        cfg = plan.resolve_config(env_key, algo, 1)          # baseline config; seed set below (not a run)
        hp = dict(cfg["algo"]["hyperparams"])
        policy = hp.pop("policy")
        hp["learning_starts"] = 10 ** 9                      # never learn
        hp["buffer_size"] = steps + 10
        venv = DummyVecEnv([lambda: build_env(cfg)])
        m = DQN(policy, venv, seed=seed, device="cpu", verbose=0, **hp)
        venv.seed(env_seed)
        eps = []
        from stable_baselines3.common.callbacks import BaseCallback

        class CB(BaseCallback):
            def _on_step(self):
                for inf in self.locals["infos"]:
                    if "episode_record" in inf:
                        eps.append(inf["episode_record"])
                return True
        m.learn(steps, callback=CB(), progress_bar=False)
        rb = m.replay_buffer
        n = rb.size()
        r, dn, to = rb.rewards[:n, 0], rb.dones[:n, 0], rb.timeouts[:n, 0]
        ends = np.flatnonzero(dn)
        starts = np.r_[0, ends[:-1] + 1]
        buf_returns = [float(r[a:b + 1].sum()) for a, b in zip(starts, ends)]
        rec_returns = [e["return_raw"] for e in eps][:len(buf_returns)]
        trunc_ok = all(bool(to[b]) == bool(e["truncated"]) for b, e in zip(ends, eps))
        return {"steps": n, "episodes": len(ends), "updates": int(m._n_updates),
                "returns_match": bool(np.allclose(buf_returns, rec_returns)),
                "timeout_flags_match_truncation": trunc_ok,
                "truncated_episodes": int(sum(e["truncated"] for e in eps)),
                "nonzero_reward_steps": int((r != 0).sum())}

    def sb3_ppo_check(algo):
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        cfg = plan.resolve_config("maintenance", algo, 1)
        cfg["env"]["kwargs"]["max_steps"] = 600            # short episodes so truncations occur inside one rollout
        hp = dict(cfg["algo"]["hyperparams"])
        policy = hp.pop("policy")
        venv = DummyVecEnv([lambda: build_env(cfg)])
        cls = RecurrentPPO if algo == "recurrent_ppo" else PPO
        m = cls(policy, venv, seed=seed, device="cpu", verbose=0, **hp)
        venv.seed(env_seed)
        eps = []

        class CB(BaseCallback):
            def _on_step(self):
                for inf in self.locals["infos"]:
                    if "episode_record" in inf:
                        eps.append(inf["episode_record"])
                return True
        _, cb = m._setup_learn(hp["n_steps"], callback=CB())
        m.collect_rollouts(m.env, cb, m.rollout_buffer, n_rollout_steps=hp["n_steps"])
        rb = m.rollout_buffer
        r = rb.rewards[:, 0]
        st = rb.episode_starts[:, 0]
        starts = np.flatnonzero(st)
        # rewards stored for truncated steps include + gamma * V(terminal obs); remove to compare raw sums
        seg = [(a, b) for a, b in zip(starts[:-1], starts[1:])]
        first_ep = [e for e in eps][:len(seg)]
        diffs = []
        for (a, b), e in zip(seg, first_ep):
            diffs.append(float(r[a:b].sum() - e["return_raw"]))
        return {"rollout_steps": int(len(r)), "complete_episodes_in_rollout": len(seg), "updates": int(m._n_updates),
                "episode_sum_minus_env_return": diffs,
                "truncated": [bool(e["truncated"]) for e in first_ep],
                "note": "a truncated episode's stored last reward includes gamma*V(terminal_obs) (SB3 timeout bootstrap)"}

    def custom_check(env_key, algo, steps=3000):
        from agents.drqn.drqn_core import DRQNAgent, DRQNConfig
        from agents.qrm.qrm_core import QRMAgent, QRMConfig
        cfg = plan.resolve_config(env_key, algo, 1)
        env = build_env(cfg)
        hp = dict(cfg["algo"]["hyperparams"])
        hp["replay_frames"] = steps + 1000
        if algo == "drqn":
            ag = DRQNAgent(env.observation_space, env.action_space.n, DRQNConfig(**hp), "cpu", RunRNG(seed))
        else:
            ag = QRMAgent(env.observation_space, env.action_space.n, env.env.rm, QRMConfig(**hp), "cpu", RunRNG(seed))
        qrm = algo != "drqn"
        obs, info = env.reset(seed=env_seed)
        head = info["rm"]["head"] if qrm else None
        ag.begin_episode(obs)
        eps = []
        for t in range(steps):
            a = ag.act(obs, head, 1.0) if qrm else ag.act(obs, 1.0)
            nobs, r, term, trunc, info = env.step(a)
            ag.observe(a, r, term, trunc, nobs, info["rm"]) if qrm else ag.observe(a, r, term, trunc, nobs)
            if term or trunc:
                eps.append(info["episode_record"])
                obs, info = env.reset()
                head = info["rm"]["head"] if qrm else None
                ag.begin_episode(obs)
            else:
                obs, head = nobs, (info["rm"]["next_head"] if qrm else None)
        rp = ag.replay
        ids = rp.episode[:rp.size]
        out_ret, out_term, out_trunc = [], [], []
        for k in range(len(eps)):
            sl = (ids == k) & rp.has_transition[:rp.size]
            out_ret.append(float(rp.reward[:rp.size][sl].sum()))
            out_term.append(bool(rp.terminated[:rp.size][sl][-1]))
            out_trunc.append(bool(rp.truncated[:rp.size][sl][-1]))
        res_ = {"steps": steps, "episodes": len(eps),
                "returns_match": bool(np.allclose(out_ret, [e["return_raw"] for e in eps])),
                "terminated_match": out_term == [e["terminated"] for e in eps],
                "truncated_match": out_trunc == [e["truncated"] for e in eps],
                "updates": ag.n_updates}
        if qrm:
            # actual head must always be a valid counterfactual head with the env's reward
            b = rp.sample_transitions(512, np.random.default_rng(env_seed))
            valid, nxt, rew, term = env.env.rm.counterfactual({k: b[k] for k in ag.EXTRA_KEYS}, b["head"], b["next_head"])
            idx = np.arange(len(b["head"]))
            res_["actual_head_valid"] = bool(valid[idx, b["head"]].all())
            res_["actual_head_reward_equals_stored"] = bool(np.allclose(rew[idx, b["head"]], b["reward"]))
            res_["actual_head_next_equals_stored"] = bool((nxt[idx, b["head"]][~term[idx, b["head"]]] ==
                                                          b["next_head"][~term[idx, b["head"]]]).all())
            res_["actual_head_terminal_equals_stored"] = bool((term[idx, b["head"]] == b["terminated"]).all())
            res_["valid_heads_per_transition"] = float(valid.sum(1).mean())
        env.close()
        return res_

    F["delivery/dqn"] = sb3_dqn_check("delivery", "dqn")
    F["delivery/dqn_ts"] = sb3_dqn_check("delivery", "dqn_ts")
    F["colour/dqn"] = sb3_dqn_check("colour", "dqn", steps=1200)
    F["delivery/drqn"] = custom_check("delivery", "drqn")
    F["delivery/qrm"] = custom_check("delivery", "qrm")
    F["colour/qrm_cue"] = custom_check("colour", "qrm_cue", steps=1500)
    F["colour/drqn"] = custom_check("colour", "drqn", steps=1500)
    F["maintenance/ppo"] = sb3_ppo_check("ppo")
    F["maintenance/ppo_rm"] = sb3_ppo_check("ppo_rm")
    F["maintenance/recurrent_ppo"] = sb3_ppo_check("recurrent_ppo")
    for k, v in F.items():
        print("F", k, v, flush=True)
    res["F_plumbing"] = F

path = os.path.join(OUT, "suspects.json")
old = read_json(path) if os.path.exists(path) else {}
old.update(res)
write_json(path, old)
print("wrote", path)
