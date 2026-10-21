"""Re-audit every accepted episode, produce an overview and deterministic preview."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from audit_goal_data import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--per-condition',type=int,default=50)
    parser.add_argument('--goal-degrees',type=float,default=180)
    args=parser.parse_args();folder=args.folder.resolve()
    manifest=json.loads((folder/'manifest.json').read_text())
    expected=4*args.per_condition;goal=args.goal_degrees
    assert len(manifest)==expected, f'Expected {expected} accepted episodes, got {len(manifest)}'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    colors=['#475569','#0284c7','#d97706','#e11d48']
    verified=[];fingerprints=set();groups={c:[] for c in range(4)}
    for entry in manifest:
        path=folder/entry['accepted_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        report=audit(path);assert report['quality_gate_pass'],report
        data=[json.loads(line) for line in path.open()];header,rows=data[0],data[1:]
        prep=folder/'preparation'/header['preparation_file']
        with prep.open() as f:
            last=None
            for line in f:last=json.loads(line)
        assert last['after']==rows[0]['before'],'Preparation/task discontinuity'
        assert header['reference_angle_rad']==rows[0]['before']['angle_rad']
        assert header['goal_angle_deg']==goal and header['rotor_position_drive'] is False
        assert rows[-1]['time_s']==len(rows)*header['control_dt_s']
        digest=hashlib.sha256(np.asarray([r['after']['q_rad']+[r['after']['angle_rad']] for r in rows],dtype='<f8').tobytes()).hexdigest()
        assert digest not in fingerprints,'Duplicate measured trajectory'
        fingerprints.add(digest)
        c=header['condition_id'];groups[c].append(report)
        angle=np.rad2deg([0]+[r['after']['angle_rad']-header['reference_angle_rad'] for r in rows])
        times=[0]+[r['time_s'] for r in rows]
        axes.flat[c].plot(times,angle,color=colors[c],alpha=.23,lw=.65)
        verified.append({**entry,**report,'accepted_file':entry['accepted_file'],'measured_state_sha256':digest})
    stats=[]
    for c in range(4):
        reports=groups[c];assert len(reports)==args.per_condition
        ax=axes.flat[c];ax.axhline(goal,color='#15803d',linestyle='--',lw=1)
        ax.set_title(f"Brake {[0,.005,.03,.1][c]:.3f} N m | {args.per_condition} / {args.per_condition} accepted")
        ax.set_xlabel('Measured task time (s)');ax.set_ylabel('Relative knob rotation (deg)');ax.set_ylim(-3,goal+5);ax.grid(alpha=.2)
        stats.append({'condition_id':c,'brake_nm':[0,.005,.03,.1][c],'count':args.per_condition,
                      'final_angle_deg_range':[min(r['final_angle_deg'] for r in reports),max(r['final_angle_deg'] for r in reports)],
                      'max_overshoot_deg':max(r['max_overshoot_deg'] for r in reports),
                      'max_reverse_deg':max(r['initial_reverse_excursion_deg'] for r in reports),
                      'max_travel_deg':max(r['angular_travel_deg'] for r in reports),
                      'max_tip_force_n':max(r['max_tip_force_n'] for r in reports),
                      'duration_s_range':[min(r['duration_s'] for r in reports),max(r['duration_s'] for r in reports)],
                      'min_final_hold_s':min(r['final_stable_hold_s'] for r in reports)})
    fig.suptitle(f'All {expected} measured trajectories | target +{goal:g} degrees | PyBullet scripted teacher')
    fig.tight_layout();fig.savefig(folder/f'all_{expected}_trajectories.png',dpi=160);fig.savefig(folder/f'all_{expected}_trajectories.pdf');plt.close(fig)
    (folder/'verification.json').write_text(json.dumps({'accepted_count':expected,'unique_measured_trajectories':len(fingerprints),'all_quality_gates_pass':True,'preparation_continuity_pass':True,'optimality_claim':False,'groups':stats,'episodes':verified},indent=2)+'\n')
    preview=folder/'preview';preview.mkdir(exist_ok=True)
    for name in ['cpu_pregrasp.json','geometry.json']:shutil.copy2(folder/name,preview/name)
    if not (preview/'assets').exists():(preview/'assets').symlink_to('../assets',target_is_directory=True)
    selection=[]
    for c in range(4):
        entry=min((e for e in manifest if e['condition_id']==c),key=lambda e:e['trial_id'])
        shutil.copy2(folder/entry['accepted_file'],preview/f'{c+1:02}.jsonl')
        selection.append(entry)
    (preview/'summary.json').write_text(json.dumps(selection,indent=2)+'\n')
    (preview/'selection.json').write_text(json.dumps({'rule':'first quality-passing seed in each condition; no selection by visual appearance','episodes':selection},indent=2)+'\n')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
