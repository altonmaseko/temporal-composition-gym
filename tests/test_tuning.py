"""Scientific invariants and failure-safe selection (no training or network)."""
import copy
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from agents.common import baseline_plan as base
from agents.common.tuning_plan import (ROOT, RESULTS, CELLS, build_plan, load_plan, candidate_config,
                                      slots, resolve_config, anchor, validate_hp)
from agents.common.tuning_records import objective, ranked, screen_report, freeze_finalists, select, manifest
from agents.common.tuning_io import trial_parent, check_path, safe_root
from agents.common.seeding import eval_episode_seeds
from agents.common.tuning_budget import remaining_seconds, projected_seconds


def test_all_baseline_hashes_and_plan_unchanged():
    saved=json.loads((Path(__file__).parent/'fixtures/baseline_hashes_pre_tuning.json').read_text())
    assert len(saved['hashes'])==440
    for key,expected in saved['hashes'].items():
        protocol,mode,env,algo,seed=key.split('/')
        assert base.config_hash(base.resolve_config(env,algo,int(seed),mode,protocol_version=protocol))==expected
    assert base.main(['write','--check'])==0


def test_frozen_baseline_manifest():
    from run_scripts.baseline.check_manifest import build_manifest
    m=build_manifest([str(ROOT/'run_scripts/baseline/results')])
    assert m['counts']=={'complete':55}
    assert m['all_complete'] and not m['unexpected_runs']


def test_plan_and_budget_rounding():
    p=load_plan()
    assert p==build_plan()
    assert len(p['cells'])==11
    hashes=set()
    for cell in p['cells']:
        assert len(cell['candidates'])==8
        assert cell['candidates'][0]['hyperparams']==anchor(cell['env'],cell['algo'])
        for c in cell['candidates']:
            cfg=c['config']; h=cfg['algo']['hyperparams']
            validate_hp(cell['env'],cell['algo'],h)
            assert cfg['budget']['env_steps']%h.get('n_steps',h.get('train_freq',1))==0
            assert cfg['budget']['env_steps']==cfg['eval']['every_steps']*20
            if 'n_steps' in h:
                assert h['n_steps']%h['batch_size']==0
            assert c['config_hash']==base.config_hash(cfg)
            hashes.add(c['config_hash'])
    assert len(hashes)==88
    ss=list(slots(p))
    assert len(ss)==154
    assert {r:sum(s['role']==r for s in ss) for r in 'abc'}=={'a':27,'b':74,'c':53}
    assert len(list(slots(p,'smoke')))==88


@pytest.mark.parametrize('seed',[1,10,100,200,201,301,401])
def test_seed_rejection(seed):
    with pytest.raises(ValueError):
        resolve_config('delivery','dqn',0,anchor('delivery','dqn'),seed)


def test_seed_separation_and_fixed_semantics():
    p=load_plan()
    for cell in p['cells']:
        e,a=cell['env'],cell['algo']
        cfg=candidate_config(p,e,a,0,199,'full')
        old=base.resolve_config(e,a,1,protocol_version='v3')
        assert cfg['env']==old['env']
        assert cfg['eval']['secondary']==old['eval']['secondary']
        assert cfg['algo']['hyperparams']['gamma']==old['algo']['hyperparams']['gamma']
        ev=cfg['eval']
        periodic=eval_episode_seeds(ev['periodic_seed_block'],ev['periodic_episodes'])
        terminal=eval_episode_seeds(ev['final_seed_block'],ev['final_episodes'])
        assert set(periodic).isdisjoint(terminal)
        assert all(3000000<=s<4000000 for s in periodic+terminal)
    bad=anchor('delivery','dqn'); bad['gamma']=.9
    with pytest.raises(ValueError):
        resolve_config('delivery','dqn',0,bad)


def test_auc_and_lexicographic_ties():
    points=[dict(global_step=s,success_rate=0,progress_mean=v,return_mean=-v) for s,v in [(0,0),(5,2),(10,4)]]
    assert objective(points,10)==(0,2,-2)
    records=[{'candidate':i,'objective':v} for i,v in enumerate([(0,2,9),(0,3,-100),(1,0,-1000),(0,2,10),(0,2,10)])]
    assert [r['candidate'] for r in ranked(records)]==[2,1,3,4,0]
    with pytest.raises(ValueError): objective(points[:-1],10)
    points[-1]['return_mean']=float('nan')
    with pytest.raises(ValueError): objective(points,10)


def test_missing_never_zero_and_override_requires_reason(tmp_path):
    p=load_plan()
    r=screen_report(tmp_path,p,'full')
    assert not r['finalists']
    assert all(v['unresolved']==list(range(8)) for v in r['cells'].values())
    assert all(x['objective'] is None for v in r['cells'].values() for x in v['screening'])
    with pytest.raises(ValueError): freeze_finalists(tmp_path,p,'full')
    with pytest.raises(ValueError): screen_report(tmp_path,p,'full',True)
    s=select(tmp_path,p,'full')
    assert not s['complete'] and not s['usable_for_final_comparison']
    m=manifest(tmp_path,p,'full')
    assert m['counts']=={'missing':88,'pending_selection':66}


def test_confirmation_selection_and_incomplete_override(tmp_path,monkeypatch):
    import agents.common.tuning_records as records
    p=load_plan()
    def fake(root,cfg,plan_hash,choices=None):
        cid,seed=cfg['tuning']['candidate'],cfg['seed']['train_seed']
        # Screening prefers 0; independent confirmations prefer 1.
        return dict(state='complete',objective=(0,8-cid if seed==101 else cid,0),config_hash=base.config_hash(cfg),attempts=[])
    monkeypatch.setattr(records,'inspect_trial',fake)
    frozen=freeze_finalists(tmp_path,p,'full')
    assert freeze_finalists(tmp_path,p,'full')['assignment_hash']==frozen['assignment_hash']
    assert all(v==[0,1] for v in frozen['finalists'].values())
    s=select(tmp_path,p,'full')
    assert s['complete'] and s['usable_for_final_comparison']
    assert all(v['candidate']==1 and v['zero_success'] and v['tie_break']=='greedy_progress_auc' for v in s['selected'].values())
    def missing(root,cfg,plan_hash,choices=None):
        if cfg['tuning']['candidate']==7:
            return dict(state='failed',objective=None,attempts=[])
        return fake(root,cfg,plan_hash,choices)
    monkeypatch.setattr(records,'inspect_trial',missing)
    assert not screen_report(tmp_path,p,'full')['finalists']
    assert len(screen_report(tmp_path,p,'full',True,'documented numerical failure')['finalists'])==11
    assert not select(tmp_path,p,'full')['complete']
    override=select(tmp_path,p,'full',True,'documented numerical failure')
    assert override['complete'] and not override['usable_for_final_comparison']


def test_portable_path_lengths():
    p=load_plan()
    for e,a in CELLS:
        for mode in ('full','smoke'):
            path=trial_parent(RESULTS,candidate_config(p,e,a,7,104,mode))/'a99'
            check_path(path,60)
            for leaf in ('checkpoints/restart.tmp/replay.pkl','metadata_resume_20261028T120000.json','config.resolved.json'):
                assert len(str(path/leaf))<240
    with pytest.raises(ValueError): check_path(ROOT/('x'*240))
    with pytest.raises(ValueError): safe_root(ROOT/'run_scripts/baseline/results/v2')
    check_path(RESULTS/'pf12345678'/'_wb'/('a'*12),90)


def test_budget_cap():
    now=datetime.now(timezone.utc)
    start=(now-timedelta(hours=1)).isoformat()
    assert remaining_seconds('b',start,now=now)==pytest.approx(32.33*3600)
    assert remaining_seconds('b',start,previously_billed=33,now=now)<0
    assert remaining_seconds('b',start,hourly_rate=.60,now=now)<17*3600
    assert projected_seconds('colour','drqn')>projected_seconds('colour','dqn')
    with pytest.raises(ValueError): remaining_seconds('b',start,hourly_rate=float('nan'),now=now)


def test_launcher_dry_run_failure_and_continue(tmp_path,monkeypatch,capsys):
    from run_scripts.tuning import launch
    slot=dict(env='delivery',algo='dqn',candidate=0,seed=101,stage='screen',role='b',slot=0)
    monkeypatch.setattr(launch,'slots',lambda *args: iter([slot]))
    args=['--role','b','--mode','smoke','--device-policy','cpu','--wandb-mode','disabled','--results-root',str(tmp_path/'r')]
    assert launch.main(args+['--dry-run'])==0
    assert not (tmp_path/'r').exists()
    class Failed:
        def __init__(self,*args,**kwargs): pass
        def wait(self,**kwargs): return 7
    monkeypatch.setattr(launch.subprocess,'Popen',Failed)
    assert launch.main(args)==1
    assert 'BATCH FAILED' in capsys.readouterr().out
    statuses=list((tmp_path/'r').rglob('status.csv'))
    assert len(statuses)==1 and ',failed,7,' in statuses[0].read_text()
    monkeypatch.setattr(launch,'inspect_trial',lambda *args: {'state':'complete'})
    # Different root avoids intentionally unique batch-directory collisions in this rapid test.
    args[-1]=str(tmp_path/'r2')
    assert launch.main(args+['--continue-batch'])==0
    assert 'SKIPPED_COMPLETE' in capsys.readouterr().out


def test_strict_record_verification_and_duplicate_detection(tmp_path):
    import csv
    from agents.common.tuning_records import verify_run,inspect_trial,expected_updates
    from agents.common.run_io import CSV_FIELDS
    p=load_plan(); cfg=candidate_config(p,'delivery','dqn',0,101,'smoke')
    path=trial_parent(tmp_path,cfg)/'a00'; path.mkdir(parents=True)
    h=base.config_hash(cfg); updates=expected_updates(cfg)
    def write(name,value): (path/name).write_text(json.dumps(value))
    write('identity.json',dict(config_hash=h,plan_hash=p['plan_hash'],phase='tuning',mode='smoke',env='delivery',algo='dqn',seed=101,candidate=0,run_id='test',stage='screen',slot=0))
    write('config.resolved.json',cfg)
    write('status.json',dict(state='complete',verified=True,final_eval_done=True,env_steps_done=64,n_updates=updates))
    summary=dict(n_episodes=1,success_rate=0,progress_mean=0,return_mean=0)
    write('final_summary.json',dict(global_step=64,n_updates=updates,config_hash=h,run_id='test',final_eval=summary,final_eval_noisy=summary))
    def csvwrite(name,rows):
        with (path/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=CSV_FIELDS[name]); w.writeheader(); w.writerows(rows)
    for suffix in ('','_noisy'):
        csvwrite('eval_points'+suffix,[dict(global_step=s,eval_index=i,**summary) for i,s in enumerate((0,32,64))])
        csvwrite('eval_periodic'+suffix,[dict(global_step=s,eval_index=i,episode_seed=3000000,success=0,progress=0,return_raw=0,length=32) for i,s in enumerate((0,32,64))])
        csvwrite('eval_final'+suffix,[dict(global_step=64,episode_seed=3010000,success=0,progress=0,return_raw=0,length=32)])
    csvwrite('train_updates',[dict(global_step=64,n_updates=updates,loss=.1)])
    assert verify_run(path,cfg,p['plan_hash'])['verified']
    import shutil
    shutil.copytree(path,path.parent/'a01')
    assert inspect_trial(tmp_path,cfg,p['plan_hash'])['state']=='duplicate'
    chosen=inspect_trial(tmp_path,cfg,p['plan_hash'],{h:path.relative_to(tmp_path).as_posix()})
    assert chosen['state']=='complete' and chosen['objective']==(0,0,0)
    m=manifest(tmp_path,p,'smoke',{h:path.relative_to(tmp_path).as_posix()})
    assert m['counts']['complete']==1 and 'duplicate' not in m['counts']
    (path/'eval_periodic_noisy.csv').unlink()
    invalid=verify_run(path,cfg,p['plan_hash'])
    assert not invalid['verified'] and invalid['objective'] is None


def test_checkpoint_replacement_retries_windows_lock(tmp_path,monkeypatch):
    from agents.common.tuning_io import TuningDirectory
    import os
    rd=TuningDirectory(str(tmp_path/'a00'))
    Path(rd.restart_dir).mkdir(parents=True)
    (Path(rd.restart_dir)/'old').write_text('old')
    temp=Path(rd.restart_dir+'.tmp'); temp.mkdir()
    (temp/'new').write_text('new')
    original=os.replace; calls=[]
    def transient(src,dst):
        calls.append(str(src))
        if len(calls)==1: raise PermissionError('transient lock')
        return original(src,dst)
    monkeypatch.setattr(os,'replace',transient)
    rd.swap_restart(str(temp))
    assert (Path(rd.restart_dir)/'new').read_text()=='new'
    assert not Path(rd.restart_dir+'.old').exists()
    rd.preserve_status({'state':'failed','error':'injected failure'})
    assert json.loads(next(Path(rd.path).glob('status_resume_*.json')).read_text())['state']=='failed'


def test_collection_excludes_checkpoints_and_sdk_cache(tmp_path):
    from run_scripts.tuning.collect import main
    import zipfile
    root=tmp_path/'r'
    folder=root/'v3/smoke/delivery/dqn/t00/s101/a00'
    for rel in ('checkpoints/restart/state.json','logs/internal.json','_wb/cache.json','status.json','resources.csv'):
        path=folder/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{}')
    out=tmp_path/'bundle.zip'
    assert main(['--results-root',str(root),'--mode','smoke','--out',str(out)])==0
    with zipfile.ZipFile(out) as z:
        names=z.namelist()
        assert len(names)==4
        assert all(Path(n).suffix in ('.csv','.json') for n in names)
        assert not any('/checkpoints/' in n or '/logs/' in n or '/_wb/' in n for n in names)
