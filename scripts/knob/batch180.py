"""Generate/audit measured target-angle episodes; no rotor goal actuation.

Scripted teacher may use resistance to select grip. This is not an ICL policy.
Preparation is retained separately; measured task starts after grasp settling.
"""
import argparse
import copy
import json
import os
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from goal_rollouts import GoalController
from cpu_rollouts import ROOT, PreviewSim, KnobGeometry, command, p, CONTROL_DT, DT
from dexx.tasks.sharpa_contact_adapt.geometry_tools import axis_rotation
from audit_goal_data import audit

GENERATOR_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class RegraspController(GoalController):
    def __init__(self, sim, pose, g, grip, rate, goal_deg=180):
        super().__init__(sim, pose, g, 'smooth', goal_deg)
        self.grip_scale=grip
        self.rate=rate
        self.stage='turn2'
        self.stage_time=0.
        self.hold=0
        self.lift=0.
        self.closed = np.array([pose.get("closed_joint_positions_rad",pose["joint_positions_rad"])[n] for n in sim.names]) if "closed_joint_positions_rad" in pose else None
        self.reference=None
        self.kp=.5;self.kd=.8;self.accel=.12
        self.quiet_hold=False;self.latched=False

    def finger_command(self,t,grip):
        if self.closed is None:return command(self.sim,t,grip)
        close=np.clip((t-.5)/1.5,0,1)*grip
        target=self.sim.initial.copy()
        target[self.sim.hand_idx]+=close*(self.closed[self.sim.hand_idx]-self.sim.initial[self.sim.hand_idx])
        return target

    def solve(self,phase,lift):
        rot=axis_rotation([0,0,1],phase)
        position=self.center+rot@(self.initial_frame[:3,3]-self.center)+[0,0,lift]
        orientation=rot@self.initial_frame[:3,:3]
        def residual(q):
            frame=self.fk(q)
            return np.r_[frame[:3,3]-position,.1*Rotation.from_matrix(orientation.T@frame[:3,:3]).as_rotvec(),.0001*(q-self.previous)]
        fit=least_squares(residual,self.previous,bounds=(self.sim.limits[self.arm,0]+.005,self.sim.limits[self.arm,1]-.005),max_nfev=30)
        self.previous=fit.x
        frame=self.fk(fit.x)
        self.ik_position_error=float(np.linalg.norm(frame[:3,3]-position))
        self.ik_orientation_error=float(np.linalg.norm(Rotation.from_matrix(orientation.T@frame[:3,:3]).as_rotvec()))
        return fit.x

    def next_stage(self,stage):
        self.stage=stage;self.stage_time=0.;self.hold=0

    def action(self,t,angle,speed):
        self.stage_time+=CONTROL_DT
        grip=self.grip_scale
        if self.stage.startswith('turn'):
            goal=np.deg2rad(90) if self.stage=='turn1' else self.goal
            self.filtered_speed+=.15*(speed-self.filtered_speed)
            req=np.clip(self.kp*(goal-angle)-self.kd*self.filtered_speed,-self.rate,self.rate)
            self.orbit_speed+=np.clip(req-self.orbit_speed,-self.accel*CONTROL_DT,self.accel*CONTROL_DT)
            if self.quiet_hold:
                if abs(goal-angle)<np.deg2rad(.15) and abs(speed)<.05:self.latched=True
                if abs(goal-angle)>np.deg2rad(.23):self.latched=False
                if self.latched:self.orbit_speed=0.
            self.phase+=self.orbit_speed*CONTROL_DT
            if self.stage=='turn1':
                self.hold=self.hold+1 if abs(goal-angle)<np.deg2rad(.25) and abs(speed)<.05 else 0
                if self.hold>=30:
                    self.release_phase=self.phase;self.next_stage('open')
        elif self.stage=='open':
            grip=self.grip_scale*(1-min(1,self.stage_time/2))
            if self.stage_time>=2:self.next_stage('lift')
        elif self.stage=='lift':
            grip=0.;self.lift=.07*min(1,self.stage_time/2)
            if self.stage_time>=2:self.next_stage('return')
        elif self.stage=='return':
            grip=0.;self.phase=self.release_phase*(1-min(1,self.stage_time/8))
            if self.stage_time>=8:self.next_stage('lower')
        elif self.stage=='lower':
            grip=0.;self.lift=.07*(1-min(1,self.stage_time/2))
            if self.stage_time>=2:self.next_stage('close')
        elif self.stage=='close':
            grip=self.grip_scale*min(1,self.stage_time/2)
            if self.stage_time>=3:
                self.orbit_speed=0.;self.filtered_speed=0.;self.next_stage('turn2')
        target=self.finger_command(2,grip)
        target[self.arm]=self.solve(self.phase,self.lift)
        return target


def run_one(spec):
    folder,condition,trial,seed,seconds,*extra=spec
    overrides=extra[0] if extra else {}
    folder=Path(folder)
    pose=json.loads((folder/'cpu_pregrasp.json').read_text())
    g=KnobGeometry(**json.loads((folder/'geometry.json').read_text()))
    brake=[0,.005,.03,.1][condition]
    rng=np.random.default_rng(seed)
    base_grip=1.0 if "closed_joint_positions_rad" in pose else (1.0 if brake<.1 else 3.0)
    grip=overrides.get("grip",base_grip)*rng.uniform(.98,1.02)
    rate=rng.uniform(.135,.15)
    p.connect(p.DIRECT)
    try:
        sim=PreviewSim((folder/'assets/preview_robot.urdf',folder/'assets/preview_knob.urdf'),pose,g,brake,'solver_friction',self_collision=True,physics_dt=overrides.get("physics_dt",1/960),solver_iterations=overrides.get("solver_iterations",150))
        goal_deg=float(overrides.get("goal_degrees",180))
        ctrl=RegraspController(sim,pose,g,grip,rate,goal_deg)
        for key in ["kp", "kd", "accel", "quiet_hold"]:
            if key in overrides:setattr(ctrl,key,overrides[key])
        before=sim.snapshot();prep=[]
        for k in range(150):
            t=(k+1)*CONTROL_DT
            after=sim.step(ctrl.finger_command(min(t,2),grip))
            prep.append({'time_s':t,'duration_s':CONTROL_DT,'before':before,'issued_target_rad':sim.target.tolist(),'after':after})
            before=after
        reference=before['angle_rad'];rows=[];hold=0;reason='timeout';travel=0.;previous_relative=0.
        for k in range(round(seconds/CONTROL_DT)):
            if (folder/"STOP").exists():
                reason="collection_stopped";break
            t=(k+1)*CONTROL_DT
            target=ctrl.action(t,before['angle_rad']-reference,before['angular_velocity_rad_s'])
            if ctrl.ik_position_error>.002 or ctrl.ik_orientation_error>.025 or abs(ctrl.phase)>np.deg2rad(300):
                reason='workspace_guard';break
            after=sim.step(target)
            rel=after['angle_rad']-reference
            good=abs(rel-ctrl.goal)<=np.deg2rad(.25) and abs(after['angular_velocity_rad_s'])<=.05
            hold=hold+1 if good else 0
            rows.append({'time_s':t,'duration_s':CONTROL_DT,'before':before,'issued_target_rad':sim.target.tolist(),'after':after,'controller_stage':ctrl.stage,'controller_wrist_orbit_rad':ctrl.phase,'ik_position_error_m':ctrl.ik_position_error,'ik_orientation_error_rad':ctrl.ik_orientation_error})
            before=after
            travel+=abs(rel-previous_relative);previous_relative=rel
            violation=max(0,float((sim.limits[:,0]-np.array(after['q_rad'])).max()),float((np.array(after['q_rad'])-sim.limits[:,1]).max()))
            if rel>ctrl.goal+np.deg2rad(1) or rel<np.deg2rad(-1) or travel>ctrl.goal+np.deg2rad(5) or after.get('max_self_penetration_m',0)>.002 or after.get('max_moving_link_table_penetration_m',0)>.002 or violation>.01 or np.linalg.norm(after['tip_normal_force_world_n'],axis=1).max()>50:
                reason='irreversible_quality_gate_failure';break
            if hold>=90:reason='stable_goal';break
        name=f'c{condition}_t{trial:03}_s{seed}'
        header={'type':'header','schema':'cpu_regrasp_goal_v2','generator_sha256':GENERATOR_HASH,'engine':'PyBullet 3.2.7','joint_names':sim.names,'robot_link_names':{str(i):n for n,i in sim.links.items()},'tactile_finger_order':['thumb','index','middle','ring','pinky'],'joint_limits_rad':sim.limits.tolist(),'brake_nm':brake,'brake_model':'solver_friction','goal_angle_deg':goal_deg,'reference_angle_rad':reference,'angle_tolerance_deg':.25,'speed_tolerance_rad_s':.05,'required_hold_s':3,'physics_dt_s':sim.physics_dt,'solver_iterations':overrides.get('solver_iterations',150),'control_dt_s':CONTROL_DT,'condition_id':condition,'trial_id':trial,'seed':seed,'controller':'wrist-orbit privileged scripted teacher','grip_scale':grip,'closure_model':'saved_pinch_targets' if ctrl.closed is not None else 'legacy_joint_offsets','max_orbit_speed_rad_s':rate,'quiet_hold':ctrl.quiet_hold,'kp':ctrl.kp,'kd':ctrl.kd,'accel':ctrl.accel,'uses_physical_condition_as_input':(overrides.get('grip',base_grip)!=1.0) if ctrl.closed is not None else True,'self_collision_enabled':True,'rotor_position_drive':False,'trained_policy':False,'termination':reason,'limitations':['CPU only; not Isaac validated','uncalibrated contact forces'],'preparation_file':name+'.preparation.jsonl'}
        path=folder/'attempts'/(name+'.jsonl')
        with path.open('x') as f:
            f.write(json.dumps(header)+'\n')
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        with (folder/'preparation'/(name+'.preparation.jsonl')).open('x') as f:
            for row in prep:f.write(json.dumps(row,allow_nan=False)+'\n')
        result=audit(path) if rows else {'quality_gate_pass':False,'rejection_reasons':['empty']}
        result.update({'condition_id':condition,'trial_id':trial,'seed':seed,'termination':reason,'file':str(path.relative_to(folder)),'success':hold>=90})
        (folder/'attempts'/(name+'.audit.json')).write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:p.disconnect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'assets/tasks/knob')
    parser.add_argument('--per-condition',type=int,default=50)
    parser.add_argument('--conditions',type=int,nargs='+',default=[0,1,2,3],choices=range(4))
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--seconds',type=float,default=65)
    parser.add_argument('--seed',type=int,default=18100)
    parser.add_argument('--goal-degrees',type=float,default=180)
    parser.add_argument('--max-attempts',type=int,default=100)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--high-grip',type=float,default=1.0)
    parser.add_argument('--high-kp',type=float,default=.5)
    parser.add_argument('--high-kd',type=float,default=.8)
    parser.add_argument('--high-accel',type=float,default=.12)
    args=parser.parse_args()
    folder=args.out.resolve()
    if not 0 < args.goal_degrees <= 180:parser.error("goal must be in (0,180]")
    import shutil,hashlib
    if not args.resume:
        folder.mkdir(parents=True,exist_ok=False)
        pose_source=args.source/'cpu_pregrasp_180.json'
        if not pose_source.exists():pose_source=args.source/'cpu_pregrasp.json'
        shutil.copy2(pose_source,folder/'cpu_pregrasp.json')
        shutil.copy2(args.source/'geometry.json',folder/'geometry.json')
        if (args.source/'assets').exists():
            shutil.copytree((args.source/'assets').resolve(),folder/'assets')
        else:
            from cpu_rollouts import prepare_urdfs
            (folder/'assets').mkdir()
            prepare_urdfs(folder/'assets',KnobGeometry(**json.loads((folder/'geometry.json').read_text())))
        # URDF mesh paths must point at this dataset's own copied meshes.
        robot=folder/'assets/preview_robot.urdf'
        import xml.etree.ElementTree as ET
        tree=ET.parse(robot)
        for mesh in tree.getroot().iter('mesh'):
            mesh.set('filename',Path(mesh.get('filename')).name)
        tree.write(robot)
        for name in ['attempts','preparation','accepted','source']:(folder/name).mkdir()
    source_snapshot=folder/'source'/('conditions_'+''.join(map(str,args.conditions))+'_'+GENERATOR_HASH[:12])
    source_snapshot.mkdir(exist_ok=True)
    for name in ['batch180.py','cpu_rollouts.py','goal_rollouts.py','audit_goal_data.py']:
        shutil.copy2(Path(__file__).parent/name,source_snapshot/name)
    config=vars(args).copy();config.update({'out':str(folder),'source':str(args.source),'quality_thresholds':{'max_overshoot_deg':1,'max_initial_reverse_deg':1,'max_extra_travel_deg':5,'hold_s':3,'final_error_deg':.25,'final_speed_rad_s':.05,'max_ik_error_m':.002,'max_joint_limit_violation_rad':.01,'max_tip_force_n':50},'variation':'seeded grip multiplier U(.98,1.02), orbit rate U(.135,.15); matched seeds across conditions','optimality_claim':False})
    (folder/('generation_config_'+''.join(map(str,args.conditions))+'_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]+'.json')).write_text(json.dumps(config,indent=2)+'\n')
    results=[json.loads(p.read_text()) for p in sorted((folder/'attempts').glob('*.audit.json'))]
    for result in results:
        with (folder/result['file']).open() as stream:header=json.loads(next(stream))
        assert header['goal_angle_deg']==args.goal_degrees, 'Cannot mix goals on resume'
        assert header['seed']==args.seed+header['trial_id'], 'Cannot change seed on resume'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            if (folder/"STOP").exists():break
            specs=[]
            for c in args.conditions:
                done=[r for r in results if r['condition_id']==c]
                passed=sum(r['quality_gate_pass'] for r in done)
                remaining=args.per_condition-passed
                if remaining<=0:continue
                first=max([r['trial_id'] for r in done],default=-1)+1
                high={'grip':args.high_grip,'kp':args.high_kp,'kd':args.high_kd,'accel':args.high_accel} if c==3 else {}
                high["goal_degrees"]=args.goal_degrees
                requested=remaining
                # After the initial sweep, compensate for measured rejection
                # rates so the final few successes do not require serial rounds.
                # Extra passing candidates remain in attempts; export the first
                # 50 by trial/seed order, never the fastest-completing futures.
                sampled=[r for r in done if r.get('termination')!='collection_stopped']
                if len(sampled)>=args.per_condition:
                    probability=(sum(r['quality_gate_pass'] for r in sampled)+1)/(len(sampled)+2)
                    requested=max(4,int(np.ceil((remaining+2*np.sqrt(remaining*(1-probability)))/probability)))
                completed_ids={r['trial_id'] for r in done}
                missing=[t for t in range(first) if t not in completed_ids]
                trial_ids=missing+list(range(first,min(args.max_attempts,first+max(0,requested-len(missing)))))
                for t in trial_ids:
                    specs.append((str(folder),c,t,args.seed+t,args.seconds,high))
            if not specs:break
            # Interleave conditions so progress can be compared during collection.
            specs.sort(key=lambda s:(s[2],s[1]))
            futures=[pool.submit(run_one,s) for s in specs]
            for future in as_completed(futures):
                result=future.result();results.append(result)
                (folder/'summary.json').write_text(json.dumps(sorted(results,key=lambda r:(r['condition_id'],r['trial_id'])),indent=2)+'\n')
                print(json.dumps(result),flush=True)
    manifest=[]
    for c in range(4):
        good=sorted([r for r in results if r['condition_id']==c and r['quality_gate_pass']],key=lambda r:r['trial_id'])[:args.per_condition]
        for i,r in enumerate(good):
            source=folder/r['file'];dest=folder/'accepted'/f'c{c}_{i:03}.jsonl'
            if dest.exists() and not os.path.samefile(source,dest):dest.unlink()
            if not dest.exists():os.link(source,dest)
            manifest.append({**r,'accepted_file':str(dest.relative_to(folder)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    counts={c:sum(r['condition_id']==c for r in manifest) for c in range(4)}
    print('Accepted counts: '+json.dumps(counts),flush=True)
    if any(counts[c]!=args.per_condition for c in args.conditions):raise SystemExit('Insufficient quality-passing episodes; failures retained.')

if __name__=='__main__':main()
