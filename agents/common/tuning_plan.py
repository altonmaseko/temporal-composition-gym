"""Frozen random search; baseline resolution remains untouched.

The checked-in plan is the input to execution, never a mutable sampler database.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from agents.common import baseline_plan as baseline
from agents.common.seeding import train_env_seed

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / 'run_scripts/tuning/trials_v3.json'
RESULTS = ROOT / 'run_scripts/tuning/results'
CELLS = [(e, a) for e, aa in (
    ('delivery', ('dqn', 'dqn_ts', 'drqn', 'qrm')),
    ('colour', ('dqn', 'dqn_ts', 'drqn', 'qrm_cue')),
    ('maintenance', ('ppo', 'recurrent_ppo', 'ppo_rm'))) for a in aa]
BUDGETS = dict(delivery=250_000, colour=750_000, maintenance=512_000)
SAMPLER_SEED = 101
CONFIRM_SEEDS = (102, 103, 104)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def spaces(env, algo):
    if env == 'maintenance':
        return {'learning_rate': {'log': [1e-5, 1e-3]}, 'ent_coef': [0.0, .001, .01, .03],
                'n_steps': [1024, 2048], 'batch_size': [64, 128], 'n_epochs': [5, 10],
                'gae_lambda': [.90, .95, .98], 'clip_range': [.1, .2, .3]}
    custom = algo in ('drqn', 'qrm', 'qrm_cue')
    s = {'lr' if custom else 'learning_rate': {'log': [1e-5, 1e-3 if env == 'delivery' else 3e-4]},
         'eps_fraction' if custom else 'exploration_fraction': [.2, .5, .8],
         'eps_end' if custom else 'exploration_final_eps': [.05, .1, .2],
         'target_update' if custom else 'target_update_interval': [1000, 2000, 5000] if env == 'delivery' else [2000, 5000, 10000],
         'train_freq': [4, 8], 'batch_size': [32, 64]}
    if algo == 'drqn':
        s['batch_size'] = [4, 8] if env == 'delivery' else [2, 4]
        s['sequence'] = [[32, 16], [64, 16]] if env == 'delivery' else [[64, 32], [96, 32]]
    return s


def anchor(env, algo):
    return baseline.resolve_config(env, algo, 1, protocol_version='v3')['algo']['hyperparams']


def validate_hp(env, algo, hp):
    base = anchor(env, algo)
    spec = spaces(env, algo)
    varying = set(spec) - {'sequence'}
    if 'sequence' in spec:
        varying |= {'train_len', 'burn_in'}
        if [hp.get('train_len'), hp.get('burn_in')] not in spec['sequence']:
            raise ValueError('invalid sequence/burn-in pair')
    if set(hp) != set(base) or any(hp[k] != base[k] for k in base if k not in varying):
        raise ValueError('task or fixed learner setting changed')
    for key, vals in spec.items():
        if key == 'sequence':
            continue
        value = hp[key]
        if isinstance(vals, dict):
            if not np.isfinite(value) or not vals['log'][0] <= value <= vals['log'][1]:
                raise ValueError(f'{key} outside log-uniform bounds')
        elif value not in vals:
            raise ValueError(f'{key} outside declared space')


def role_for(env, algo, slot):
    if env == 'colour':
        return ('a' if slot < 7 else 'b') if algo == 'drqn' else 'c'
    if env == 'delivery':
        return ('a' if slot < 6 else 'b') if algo == 'drqn' else 'b'
    return 'a' if algo == 'recurrent_ppo' else 'b' if algo == 'ppo' or slot < 3 else 'c'


def resolve_config(env, algo, candidate, hp, seed=101, mode='full'):
    if (env, algo) not in CELLS or mode not in ('full', 'smoke') or seed not in range(101, 200):
        raise ValueError('invalid tuning cell, mode or seed (101-199 required)')
    if not 0 <= candidate < 8:
        raise ValueError('candidate must be 0..7')
    validate_hp(env, algo, hp)
    cfg = baseline.resolve_config(env, algo, 1, protocol_version='v3')
    cfg['phase'], cfg['mode'] = 'tuning', mode
    cfg['algo']['hyperparams'] = copy.deepcopy(hp)
    cfg['seed'] = {'train_seed': seed, 'train_env_reset_seed': train_env_seed(seed)}
    cfg['budget']['env_steps'] = b = BUDGETS[env]
    cfg['eval'].update(every_steps=b // 20, periodic_seed_block='tuning_validation',
                       final_seed_block='tuning_terminal')
    cfg['checkpoint']['every_steps'] = b // 10
    cfg['logging']['train_log_every_steps'] = b // 200
    cfg['wandb']['group'] = 'tuning-v3' + ('-smoke' if mode == 'smoke' else '')
    cfg['tuning'] = {'candidate': candidate, 'learner_hash': digest(hp), 'schema': 1}
    if mode == 'smoke':
        # Plumbing only: two candidates/cell, short episodes and tiny replay/updates.
        cfg['budget']['env_steps'] = b = 64
        cfg['env']['kwargs']['max_steps'] = 32
        cfg['eval'].update(n_points=2, every_steps=32, periodic_episodes=1, final_episodes=1)
        cfg['checkpoint']['every_steps'] = 32
        cfg['logging']['train_log_every_steps'] = 16
        h = cfg['algo']['hyperparams']
        if env == 'maintenance':
            h.update(n_steps=32, batch_size=16, n_epochs=1)
        elif algo in ('dqn', 'dqn_ts'):
            h.update(buffer_size=128, learning_starts=32, batch_size=8, target_update_interval=32, train_freq=8)
        else:
            h.update(replay_frames=128, learning_starts=32, batch_size=2, target_update=32, train_freq=8)
            if algo == 'drqn':
                h.update(train_len=8, burn_in=4, batch_size=1)
    baseline.validate_config(cfg)
    return cfg


def build_plan():
    cells = []
    for index, (env, algo) in enumerate(CELLS):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(SAMPLER_SEED, spawn_key=(index,))))
        candidates, seen = [], set()
        while len(candidates) < 8:
            hp = anchor(env, algo)
            if candidates:
                for k, values in spaces(env, algo).items():
                    value = float(np.exp(rng.uniform(*np.log(values['log'])))) if isinstance(values, dict) else copy.deepcopy(values[int(rng.integers(len(values)))])
                    if k == 'sequence':
                        hp['train_len'], hp['burn_in'] = value
                    else:
                        hp[k] = value
            if digest(hp) in seen:
                continue
            seen.add(digest(hp))
            cid = len(candidates)
            cfg = resolve_config(env, algo, cid, hp)
            candidates.append({'candidate': cid, 'hyperparams': hp, 'config': cfg,
                               'config_hash': baseline.config_hash(cfg)})
        cells.append({'env': env, 'algo': algo, 'stream': index, 'spaces': spaces(env, algo),
                      'candidates': candidates, 'slot_roles': [role_for(env, algo, i) for i in range(14)]})
    plan = {'schema': 1, 'protocol_version': 'v3', 'phase': 'tuning', 'sampler': 'numpy.PCG64',
            'sampler_seed': SAMPLER_SEED, 'screen_seed': 101, 'confirmation_seeds': list(CONFIRM_SEEDS),
            'full_runs': 154, 'smoke_runs': 88, 'cells': cells}
    plan['plan_hash'] = digest(plan)
    return plan


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan != build_plan():
        raise ValueError('trial list differs from frozen generator; run generate_trials.py --check')
    return plan


def candidate_config(plan, env, algo, candidate, seed, mode):
    cell = next(c for c in plan['cells'] if (c['env'], c['algo']) == (env, algo))
    return resolve_config(env, algo, candidate, cell['candidates'][candidate]['hyperparams'], seed, mode)


def screen_count(mode):
    return 8 if mode == 'full' else 2


def slots(plan, mode='full', finalists=None):
    for cell in plan['cells']:
        e, a = cell['env'], cell['algo']
        for cid in range(screen_count(mode)):
            yield {'env': e, 'algo': a, 'candidate': cid, 'seed': 101, 'stage': 'screen',
                   'slot': cid, 'role': cell['slot_roles'][cid]}
        selected = (finalists or {}).get(f'{e}/{a}', [])
        for rank in range(2):
            for i, seed in enumerate(CONFIRM_SEEDS):
                slot = 8 + rank * 3 + i
                yield {'env': e, 'algo': a, 'candidate': selected[rank] if len(selected) > rank else None,
                       'seed': seed, 'stage': 'confirm', 'rank': rank, 'slot': slot, 'role': cell['slot_roles'][slot]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--out', type=Path, default=PLAN_PATH)
    args = p.parse_args(argv)
    plan = build_plan()
    if args.check:
        load_plan(args.out)
        print('trials_v3.json matches frozen generator (88 candidates, 66 confirmation slots)')
    else:
        if args.out.exists() and json.loads(args.out.read_text()) != plan:
            raise ValueError('refusing to overwrite a different trial list')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
