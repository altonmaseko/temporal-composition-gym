"""Verify trial-owned records, rank AUCs, and freeze finalist assignments."""
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from agents.common import baseline_plan as base
from agents.common.run_io import read_json, write_json_atomic
from agents.common.seeding import eval_episode_seeds
from agents.common.tuning_io import check_path, safe_root, trial_parent
from agents.common.tuning_plan import candidate_config, digest, screen_count, slots


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite metric')
    return value


def objective(points, budget):
    steps = [number(p['global_step']) for p in points]
    if len(steps) < 2 or steps[0] != 0 or steps[-1] != budget or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError('incomplete or unordered AUC curve')
    return tuple(sum((steps[i] - steps[i-1]) * (number(points[i][key]) + number(points[i-1][key])) / 2
                     for i in range(1, len(points))) / budget
                 for key in ('success_rate', 'progress_mean', 'return_mean'))


def ranked(records):
    return sorted(records, key=lambda r: (*(-v for v in r['objective']), r['candidate']))


def expected_updates(cfg):
    b, h, a = cfg['budget']['env_steps'], cfg['algo']['hyperparams'], cfg['algo']['name']
    if cfg['env']['key'] == 'maintenance':
        return (b // h['n_steps']) * h['n_epochs']
    if a in ('dqn', 'dqn_ts'):
        return (b - h['learning_starts']) // h['train_freq']
    return b // h['train_freq'] - (h['learning_starts'] - 1) // h['train_freq']


def verify_run(path, cfg, plan_hash, assignment_hash=None, slot=None):
    """Strict independent verification; an invalid run has no objective."""
    path = Path(path)
    result = {'run_dir': str(path), 'verified': False, 'objective': None, 'problems': []}
    try:
        ident, actual, status = [read_json(str(path / n)) for n in ('identity.json', 'config.resolved.json', 'status.json')]
        result['state'] = status.get('state', 'unknown')
        h = base.config_hash(cfg)
        if base.config_hash(actual) != h or ident['config_hash'] != h or ident.get('plan_hash') != plan_hash:
            raise ValueError('configuration/plan hash mismatch')
        if any(ident.get(k) != v for k, v in {'phase': 'tuning', 'mode': cfg['mode'], 'env': cfg['env']['key'],
                'algo': cfg['algo']['name'], 'seed': cfg['seed']['train_seed'], 'candidate': cfg['tuning']['candidate']}.items()):
            raise ValueError('identity mismatch')
        confirming = cfg['seed']['train_seed'] != 101
        if ident.get('stage') != ('confirm' if confirming else 'screen'):
            raise ValueError('trial stage mismatch')
        if confirming and (assignment_hash is None or ident.get('assignment_hash') != assignment_hash):
            raise ValueError('confirmation assignment mismatch')
        expected_slot = slot if confirming else cfg['tuning']['candidate']
        if ident.get('slot') != expected_slot:
            raise ValueError('registered slot mismatch')
        if status.get('state') != 'complete' or status.get('verified') is not True:
            raise ValueError(f"run status {status.get('state')}: {status.get('error')}")
        summary = read_json(str(path / 'final_summary.json'))
        budget, ev = cfg['budget']['env_steps'], cfg['eval']
        updates = expected_updates(cfg)
        if not (status.get('final_eval_done') is True and status['env_steps_done'] == summary['global_step'] == budget and
                status['n_updates'] == summary['n_updates'] == updates and updates > 0 and
                summary['config_hash'] == h and summary['run_id'] == ident['run_id']):
            raise ValueError('budget, update count or summary identity mismatch')
        expected_steps = list(range(0, budget + 1, ev['every_steps']))
        curves = {}
        for suffix in ('', '_noisy'):
            points = read_rows(path / f'eval_points{suffix}.csv')
            if [int(r['global_step']) for r in points] != expected_steps or [int(r['eval_index']) for r in points] != list(range(ev['n_points'] + 1)):
                raise ValueError('missing/duplicate/misplaced evaluation point')
            per = read_rows(path / f'eval_periodic{suffix}.csv')
            seeds = eval_episode_seeds(ev['periodic_seed_block'], ev['periodic_episodes'])
            expected = [(i, step, seed) for i, step in enumerate(expected_steps) for seed in seeds]
            if [(int(r['eval_index']), int(r['global_step']), int(r['episode_seed'])) for r in per] != expected:
                raise ValueError('periodic episode count/seeds/cadence mismatch')
            for i, point in enumerate(points):
                rows = per[i * len(seeds):(i+1) * len(seeds)]
                if int(point['n_episodes']) != len(seeds):
                    raise ValueError('incorrect point episode count')
                for out, raw in (('success_rate', 'success'), ('progress_mean', 'progress'), ('return_mean', 'return_raw')):
                    mean = statistics.mean(number(r[raw]) for r in rows)
                    if not math.isclose(number(point[out]), mean, rel_tol=1e-10, abs_tol=1e-8):
                        raise ValueError('point summary differs from episode records')
            final = read_rows(path / f'eval_final{suffix}.csv')
            if [int(r['episode_seed']) for r in final] != eval_episode_seeds(ev['final_seed_block'], ev['final_episodes']) or any(int(r['global_step']) != budget for r in final):
                raise ValueError('terminal validation count/seeds/step mismatch')
            final_summary = summary['final_eval' + suffix]
            if final_summary['n_episodes'] != len(final):
                raise ValueError('terminal summary count mismatch')
            for out, raw in (('success_rate', 'success'), ('progress_mean', 'progress'), ('return_mean', 'return_raw')):
                if not math.isclose(number(final_summary[out]), statistics.mean(number(r[raw]) for r in final), rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError('terminal summary differs from records')
            for row in per + final:
                if number(row['success']) not in (0, 1) or number(row['progress']) < 0 or not 0 < number(row['length']) <= cfg['env']['kwargs']['max_steps']:
                    raise ValueError('invalid episode metrics')
            curves[suffix] = objective(points, budget)
        logs = read_rows(path / 'train_updates.csv')
        if not logs or int(logs[-1]['n_updates']) != updates:
            raise ValueError('updates log incomplete')
        for row in logs:
            for key in ('loss', 'q_mean', 'grad_norm'):
                if row.get(key):
                    number(row[key])
        if not any(r.get('loss') for r in logs):
            raise ValueError('no measured learner loss')
        result.update(verified=True, state='complete', objective=curves[''], noisy_objective=curves['_noisy'],
                      config_hash=h, n_updates=updates, final_success=summary['final_eval']['success_rate'])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result['problems'].append(str(exc))
    return result


def inspect_trial(root, cfg, plan_hash, choices=None):
    root = safe_root(root)
    assignment_hash, slot = None, None
    if cfg['seed']['train_seed'] != 101:
        try:
            assignment = read_json(str(root/'v3'/cfg['mode']/'finalists.json'))
            assignment_hash = assignment['assignment_hash']
            if (assignment.get('plan_hash') != plan_hash or
                    assignment_hash != digest({k:v for k,v in assignment.items() if k != 'assignment_hash'})):
                raise ValueError('invalid confirmation assignment')
            rank = assignment['finalists'][f"{cfg['env']['key']}/{cfg['algo']['name']}"].index(cfg['tuning']['candidate'])
            if cfg['seed']['train_seed'] not in (102,103,104):
                raise ValueError('unregistered confirmation seed')
            slot = 8 + rank * 3 + cfg['seed']['train_seed'] - 102
        except (OSError,ValueError,KeyError,TypeError) as exc:
            return {'state':'invalid_assignment','attempts':[],'objective':None,'problems':[str(exc)]}
    attempts = []
    for path in sorted(trial_parent(root, cfg).glob('a*')):
        if path.is_dir():
            rec = verify_run(path, cfg, plan_hash, assignment_hash, slot)
            rec['run_dir'] = path.relative_to(root).as_posix()
            attempts.append(rec)
    good = [r for r in attempts if r['verified']]
    selected_path = (choices or {}).get(base.config_hash(cfg))
    if selected_path:
        good = [r for r in good if r['run_dir'] == selected_path]
        if len(good) != 1:
            return {'state': 'invalid_choice', 'attempts': attempts, 'objective': None}
    if len(good) == 1:
        return {**good[0], 'attempts': attempts, 'explicit_attempt_choice': selected_path}
    state = 'duplicate' if good else 'missing' if not attempts else 'failed'
    if not good and any(r.get('state') in ('running', 'interrupted') for r in attempts):
        state = 'unfinished'
    return {'state': state, 'attempts': attempts, 'objective': None}


def screen_report(root, plan, mode, allow_incomplete=False, reason=None, choices=None):
    if allow_incomplete and not reason:
        raise ValueError('--allow-incomplete requires --reason')
    cells, finalists = {}, {}
    for cell in plan['cells']:
        e, a = cell['env'], cell['algo']
        rows = []
        for cid in range(screen_count(mode)):
            cfg = candidate_config(plan, e, a, cid, 101, mode)
            rows.append({'candidate': cid, **inspect_trial(root, cfg, plan['plan_hash'], choices)})
        unresolved = [r['candidate'] for r in rows if r['state'] != 'complete']
        good = ranked([r for r in rows if r['state'] == 'complete'])
        ready = len(good) >= 2 and (not unresolved or allow_incomplete)
        cells[f'{e}/{a}'] = {'screening': rows, 'unresolved': unresolved, 'ready': ready}
        if ready:
            finalists[f'{e}/{a}'] = [r['candidate'] for r in good[:2]]
    return {'plan_hash': plan['plan_hash'], 'mode': mode, 'allow_incomplete': allow_incomplete,
            'reason': reason, 'cells': cells, 'finalists': finalists}


def freeze_finalists(root, plan, mode, allow_incomplete=False, reason=None, choices=None):
    report = screen_report(root, plan, mode, allow_incomplete, reason, choices)
    if len(report['finalists']) != len(plan['cells']):
        raise ValueError('screening incomplete: need two valid candidates and resolved trials in every cell')
    # Relative evidence is portable when copying machine records into a merged root.
    path = safe_root(root) / 'v3' / mode / 'finalists.json'
    value = {'schema': 1, **report}
    value['assignment_hash'] = digest(value)
    if path.exists():
        old = read_json(str(path))
        if digest(old) != digest(value):
            raise ValueError('frozen finalist evidence changed; do not silently reassign confirmation slots')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(str(check_path(path, 20)), value)
    return value


def load_finalists(root, plan, mode):
    path = safe_root(root) / 'v3' / mode / 'finalists.json'
    if not path.exists():
        return None
    value = read_json(str(path))
    if value.get('plan_hash') != plan['plan_hash'] or value.get('mode') != mode or value.get('assignment_hash') != digest({k:v for k,v in value.items() if k != 'assignment_hash'}):
        raise ValueError('invalid finalist assignment')
    for ids in value['finalists'].values():
        if len(ids) != 2 or len(set(ids)) != 2 or any(i not in range(screen_count(mode)) for i in ids):
            raise ValueError('invalid finalist candidates')
    return value


def manifest(root, plan, mode, choices=None):
    root = safe_root(root)
    assignment = load_finalists(root, plan, mode)
    entries = []
    for slot in slots(plan, mode, assignment['finalists'] if assignment else None):
        if slot['candidate'] is None:
            rec = {'state': 'pending_selection', 'objective': None, 'attempts': []}
        else:
            cfg = candidate_config(plan, slot['env'], slot['algo'], slot['candidate'], slot['seed'], mode)
            rec = inspect_trial(root, cfg, plan['plan_hash'], choices)
        entries.append({**slot, **rec})
    by_role = {role: dict(Counter(r['state'] for r in entries if r['role'] == role)) for role in ('a','b','c')}
    expected_paths = {str(trial_parent(root, candidate_config(plan, r['env'], r['algo'], r['candidate'], r['seed'], mode)).resolve())
                      for r in entries if r['candidate'] is not None}
    unexpected = [str(p.relative_to(root)) for p in (safe_root(root)/'v3'/mode).glob('*/*/t*/s*/a*/identity.json')
                  if str(p.parent.parent.resolve()) not in expected_paths]
    return {'protocol_version': 'v3', 'phase': 'tuning', 'mode': mode, 'plan_hash': plan['plan_hash'],
            'expected': len(entries), 'counts': dict(Counter(r['state'] for r in entries)), 'by_role': by_role,
            'entries': entries, 'unexpected_runs': unexpected,
            'all_complete': all(r['state'] == 'complete' for r in entries) and not unexpected}


def select(root, plan, mode, allow_incomplete=False, reason=None, choices=None):
    screen = screen_report(root, plan, mode, allow_incomplete, reason, choices)
    assignment = load_finalists(root, plan, mode)
    selected, problems, confirmation = {}, [], {}
    if not assignment:
        problems.append('finalists.json missing; prepare confirmation first')
    elif assignment['finalists'] != screen['finalists']:
        problems.append('screening no longer agrees with frozen finalists')
    if assignment and assignment.get('allow_incomplete') and not allow_incomplete:
        problems.append('finalists were selected with incomplete screening; explicit override required')
    if not problems:
        for cell in plan['cells']:
            e, a = cell['env'], cell['algo']
            key = f'{e}/{a}'
            finalists = []
            confirmation[key] = []
            for cid in assignment['finalists'][key]:
                evidence = []
                for seed in (102, 103, 104):
                    cfg = candidate_config(plan, e, a, cid, seed, mode)
                    evidence.append({'seed': seed, **inspect_trial(root, cfg, plan['plan_hash'], choices)})
                confirmation[key].append({'candidate': cid, 'evidence': evidence})
                if all(r['state'] == 'complete' for r in evidence):
                    tuples = [r['objective'] for r in evidence]
                    noisy = [r['noisy_objective'] for r in evidence if r.get('noisy_objective') is not None]
                    finalists.append({'candidate': cid, 'objective': tuple(statistics.mean(v[i] for v in tuples) for i in range(3)),
                                      'noisy_mean': tuple(statistics.mean(v[i] for v in noisy) for i in range(3)) if len(noisy) == 3 else None,
                                      'std': tuple(statistics.stdev(v[i] for v in tuples) for i in range(3)), 'evidence': evidence})
            if len(finalists) != 2:
                problems.append(f'{key}: both finalists require all three confirmation seeds')
                continue
            ordered = ranked(finalists)
            win, other = ordered
            fields = ('greedy_success_auc', 'greedy_progress_auc', 'greedy_return_auc')
            tie_break = next((name for i, name in enumerate(fields) if win['objective'][i] != other['objective'][i]), 'candidate_id')
            selected[key] = {'status': 'selected', **win, 'hyperparams': cell['candidates'][win['candidate']]['hyperparams'],
                             'screening': screen['cells'][key], 'finalists': ordered, 'tie_break': tie_break,
                             'zero_success': win['objective'][0] == 0,
                             'config_hashes': [r['config_hash'] for r in win['evidence']]}
    value = {'schema': 1, 'phase': 'tuning', 'protocol_version': 'v3', 'mode': mode, 'plan_hash': plan['plan_hash'],
             'allow_incomplete': allow_incomplete, 'override_reason': reason, 'problems': problems,
             'complete': not problems and len(selected) == 11, 'selected': selected, 'screening': screen, 'confirmation': confirmation,
             'usable_for_final_comparison': mode == 'full' and not problems and not allow_incomplete}
    out = safe_root(root)/'v3'/mode
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(check_path(out/'selected_configs.json', 20)), value)
    lines = [f'# Tuning v3 selection ({mode})', '', f"Plan: `{plan['plan_hash']}`", '',
             f'Complete: {value["complete"]}. Incomplete-data override: {allow_incomplete}; reason: {reason}.',
             'Smoke outputs cannot be used for scientific selection.' if mode == 'smoke' else 'Selection uses confirmation seeds 102–104 only.', '',
             '| Cell | Candidate | Greedy success / progress / return AUC | Seed SD | Noisy AUC (report only) | Tie-break |', '|---|---|---|---|---|---|']
    for key, r in selected.items():
        lines.append(f"| {key} | {r['candidate']:02} | {r['objective']} | {r['std']} | {r['noisy_mean']} | {r['tie_break']} |")
    lines += ['', 'Unavailable results are never zero. Per-seed variability, attempts and screening failures are recorded in selected_configs.json.', '', *problems]
    check_path(out/'selection_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return value
