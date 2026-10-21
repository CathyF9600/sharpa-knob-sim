"""Export measured CPU episodes to synchronized, calibrated RGB-D HDF5.

No physics stepping during rendering. Metadata and privileged labels are separate
from observations. RGB uses the depth optical frame (synthetic aligned RGB-D),
not the unknown RealSense color-camera calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import h5py
import numpy as np
from cpu_rollouts import ROOT,PreviewSim,KnobGeometry,p
from visual_markings import Markings,build
from dexx.deploy_config import ARM_BASE_POS,SIM_INTRINSICS,DEPTH_W,DEPTH_H,TABLE_SURFACE_Z

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def camera_setup(folder):
    folder=Path(folder);cal=folder/'camera';cal.mkdir(exist_ok=True)
    source=ROOT/'calib/camera_align/current.npy';dest=cal/'camera_in_armbase.npy'
    if not dest.exists():dest.write_bytes(source.read_bytes())
    assert sha(dest)==sha(source),'Camera calibration changed during export'
    T=np.load(dest).astype(float);assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
    world=T.copy();world[:3,3]+=ARM_BASE_POS
    K=np.array([[SIM_INTRINSICS['fx'],0,SIM_INTRINSICS['cx']],[0,SIM_INTRINSICS['fy'],SIM_INTRINSICS['cy']],[0,0,1]])
    cfg={'width':DEPTH_W,'height':DEPTH_H,'fps':30,'K':K.tolist(),'T_world_camera_optical':world.tolist(),'T_armbase_camera_optical':T.tolist(),'source_extrinsic':'calib/camera_align/current.npy','source_extrinsic_sha256':sha(source),'arm_base_world_m':list(ARM_BASE_POS),'optical_frame':'x right, y down, z forward','near_m':.01,'far_m':3.,'depth_definition':'optical-axis Z in metres; invalid/background = 0','rgb_definition':'synthetic RGB aligned exactly to depth optical camera; not real color-camera intrinsics','renderer':'PyBullet TinyRenderer','distortion':'none; ideal pinhole','sensor_noise':'none; not RealSense sensor simulation','full_robot_visible':True,'visual_markings':True,'training_overlay':False}
    if (cal/'camera.json').exists():assert json.loads((cal/'camera.json').read_text())==cfg,'Camera configuration drift'
    (cal/'camera.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return cfg

def matrices(cfg):
    T=np.array(cfg['T_world_camera_optical']);eye=T[:3,3]
    view=p.computeViewMatrix(eye,eye+T[:3,2],-T[:3,1]);K=np.array(cfg['K']);fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2];n=cfg['near_m'];w,h=cfg['width'],cfg['height']
    # TinyRenderer samples at (u, v+1); shift projection so exported pixels
    # follow Dex-X's optical rays at (u+0.5, v+0.5). K remains unchanged.
    cx-=.5;cy+=.5
    proj=p.computeProjectionMatrix(-cx/fx*n,(w-cx)/fx*n,-(h-cy)/fy*n,cy/fy*n,n,cfg['far_m'])
    return view,proj

def backproject(depth,cfg):
    K=np.array(cfg['K']);u,v=np.meshgrid(np.arange(cfg['width'])+.5,np.arange(cfg['height'])+.5)
    xyz=np.stack(((u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth),axis=-1)
    T=np.array(cfg['T_world_camera_optical']);return xyz@T[:3,:3].T+T[:3,3]

def make_sim(folder,header):
    pose=json.loads((folder/'cpu_pregrasp.json').read_text());g=KnobGeometry(**json.loads((folder/'geometry.json').read_text()))
    sim=PreviewSim((folder/'assets/preview_robot.urdf',folder/'assets/preview_knob.urdf'),pose,g,header['brake_nm'],'solver_friction')
    assert sim.names==header['joint_names']
    p.changeVisualShape(sim.knob,0,rgbaColor=[0,0,0,0])
    marks=Markings(folder/'assets/markings',np.array(pose['fixture_position_m'])+[0,0,g.rotor_center_z_m])
    return sim,marks

def render(sim,marks,state,cfg,view,proj):
    for joint,q in zip(sim.ids,state['q_rad']):p.resetJointState(sim.robot,joint,q)
    p.resetJointState(sim.knob,0,state['angle_rad']);marks.update(state['angle_rad'])
    result=p.getCameraImage(cfg['width'],cfg['height'],view,proj,renderer=p.ER_TINY_RENDERER,shadow=0,lightDirection=[-1,-1,3])
    rgb=np.asarray(result[2],dtype=np.uint8)[:,:,:3];z=np.asarray(result[3],dtype=np.float32)
    valid=(z<1.-1e-6)&(z>=0);near,far=cfg['near_m'],cfg['far_m'];depth=np.where(valid,far*near/(far-(far-near)*z),0).astype(np.float32)
    seg=np.asarray(result[4]);return rgb,depth,valid.astype(np.uint8),seg

def export_one(spec):
    folder,entry=spec;folder=Path(folder);source=folder/entry['accepted_file'];name=source.stem;dest=folder/'episodes'/(name+'.h5');reportfile=dest.with_suffix('.audit.json');source_sha=sha(source)
    if dest.exists() and reportfile.exists():
        report=json.loads(reportfile.read_text())
        assert report['source_sha256']==source_sha and report['h5_sha256']==sha(dest)
        return report
    cfg=json.loads((folder/'camera/camera.json').read_text());view,proj=matrices(cfg)
    data=[json.loads(s) for s in source.open()];header,rows=data[0],data[1:];states=[rows[0]['before']]+[r['after'] for r in rows];N=len(rows);assert N>0
    with (folder/'preparation'/header['preparation_file']).open() as stream:
        for line in stream:prep_last=json.loads(line)
    assert prep_last['after']==states[0]
    tmp=dest.with_suffix('.h5.partial');p.connect(p.DIRECT)
    try:
        sim,marks=make_sim(folder,header);digests=[];visible=[];frame_checks={};table_errors=[]
        with h5py.File(tmp,'w') as f:
            f.attrs['schema']='dexx_camera_cpu_rgbd_demo_v1';f.attrs['complete']=False;f.attrs['source_sha256']=source_sha
            f.attrs['source_file']=entry['accepted_file'];f.attrs['header_json']=json.dumps(header);f.attrs['camera_json']=json.dumps(cfg)
            f.attrs['alignment']='obs[t] before actions[t]; obs[t+1] after actions[t]; final observation has no outgoing action'
            f.attrs['instruction']='Rotate the knob counterclockwise by 90 degrees from its settled starting angle.'
            f.attrs['joint_names_json']=json.dumps(header['joint_names']);f.attrs['tactile_finger_order_json']=json.dumps(header['tactile_finger_order'])
            obs=f.create_group('observations');act=f.create_group('actions');priv=f.create_group('privileged')
            frames=N+1;h,w=cfg['height'],cfg['width']
            rgbds=obs.create_dataset('rgb',(frames,h,w,3),dtype='u1',chunks=(1,h,w,3),compression='gzip',compression_opts=1)
            dds=obs.create_dataset('depth_m',(frames,h,w),dtype='<f4',chunks=(1,h,w),compression='gzip',compression_opts=1,shuffle=True)
            vds=obs.create_dataset('depth_valid',(frames,h,w),dtype='u1',chunks=(1,h,w),compression='gzip',compression_opts=1)
            for key in ['q_rad','dq_rad_s','tip_normal_force_world_n']:obs.create_dataset(key,data=np.asarray([s[key] for s in states],dtype='<f8'))
            obs.create_dataset('previous_target_rad',data=np.asarray([prep_last['issued_target_rad']]+[r['issued_target_rad'] for r in rows],dtype='<f8'))
            obs.create_dataset('timestamp_s',data=np.asarray([0]+[r['time_s'] for r in rows],dtype='<f8'))
            obs.create_dataset('goal_relative_angle_rad',data=np.full(frames,np.deg2rad(header['goal_angle_deg'])))
            act.create_dataset('joint_position_target_rad',data=np.asarray([r['issued_target_rad'] for r in rows],dtype='<f8'))
            act.create_dataset('timestamp_s',data=np.arange(N)*header['control_dt_s'])
            terminal=np.zeros(N,dtype=bool);terminal[-1]=True;f.create_dataset('transition_terminal',data=terminal)
            priv.create_dataset('rotor_angle_rad',data=[s['angle_rad'] for s in states]);priv.create_dataset('rotor_relative_angle_rad',data=[s['angle_rad']-header['reference_angle_rad'] for s in states]);priv.create_dataset('rotor_velocity_rad_s',data=[s['angular_velocity_rad_s'] for s in states]);priv.attrs['brake_nm']=header['brake_nm'];priv.attrs['condition_id']=header['condition_id'];priv.attrs['seed']=header['seed']
            for i,state in enumerate(states):
                rgb,depth,valid,seg=render(sim,marks,state,cfg,view,proj);rgbds[i]=rgb;dds[i]=depth;vds[i]=valid
                assert np.isfinite(depth).all() and np.all(depth[valid.astype(bool)]>0)
                digests.append(np.frombuffer(hashlib.sha256(rgb.tobytes()+depth.tobytes()).digest(),dtype=np.uint8))
                objects=seg&((1<<24)-1);visible.append(int(np.isin(objects,marks.bodies).sum()))
                if i in [0,N//2,N]:
                    mask=objects==sim.table;world=backproject(depth,cfg)
                    # Table top pixels away from silhouette; mesh top z is known.
                    inner=mask.copy();inner[1:-1,1:-1]&=mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:]
                    points=world[inner & valid.astype(bool)];points=points[np.abs(points[:,2]-TABLE_SURFACE_Z)<.01]
                    assert len(points)>100,'Insufficient visible table for depth geometry check'
                    error=float(np.quantile(np.abs(points[:,2]-TABLE_SURFACE_Z),.99));assert error<.002,error;table_errors.append(error)
                    frame_checks[str(i)]={'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'depth_sha256':hashlib.sha256(depth.tobytes()).hexdigest()}
            priv.create_dataset('visible_rotor_scale_pixels',data=visible)
            f.create_dataset('frame_rgb_depth_sha256',data=np.asarray(digests))
            f.attrs['complete']=True
        # Verify disk readback and independently replay three stored timestamps.
        with h5py.File(tmp,'r') as f:
            assert f['observations/rgb'].shape[0]==f['actions/joint_position_target_rad'].shape[0]+1
            for i in [0,N//2,N]:
                rgb,depth,valid,_=render(sim,marks,states[i],cfg,view,proj)
                assert np.array_equal(f['observations/rgb'][i],rgb)
                assert np.array_equal(f['observations/depth_m'][i],depth)
                assert np.array_equal(f['observations/depth_valid'][i],valid)
            assert np.array_equal(f['actions/joint_position_target_rad'][:],np.array([r['issued_target_rad'] for r in rows]))
            assert np.array_equal(f['observations/q_rad'][:],np.array([s['q_rad'] for s in states]))
        tmp.replace(dest)
        report={'episode':name,'file':str(dest.relative_to(folder)),'condition_id':header['condition_id'],'seed':header['seed'],'frames':N+1,'transitions':N,'duration_s':rows[-1]['time_s'],'source_sha256':source_sha,'h5_sha256':sha(dest),'all_frames_finite_depth':True,'alignment_verified':True,'replay_readback_verified':True,'table_depth_99percentile_error_max_m':max(table_errors),'frame_spot_checks':frame_checks,'zero_visible_rotor_scale_frames':visible.count(0),'minimum_visible_rotor_scale_pixels':min(visible),'bytes':dest.stat().st_size}
        reportfile.write_text(json.dumps(report,indent=2)+'\n');return report
    finally:p.disconnect()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--workers',type=int,default=12);parser.add_argument('--limit',type=int);args=parser.parse_args();folder=args.folder.resolve()
    camera_setup(folder);build(folder/'assets/markings');(folder/'episodes').mkdir(exist_ok=True)
    source=json.loads((folder/'manifest.json').read_text());source=source[:args.limit] if args.limit else source
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(export_one,(str(folder),e)) for e in source]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            tmp=folder/'visual_progress.tmp';tmp.write_text(json.dumps(sorted(results,key=lambda r:r['episode']),indent=2)+'\n');tmp.replace(folder/'visual_progress.json')
            print(json.dumps({k:result[k] for k in ['episode','frames','bytes','zero_visible_rotor_scale_frames']}),flush=True)
    (folder/'visual_manifest.json').write_text(json.dumps(sorted(results,key=lambda r:r['episode']),indent=2)+'\n')
    print('RGBD export complete:',len(results),flush=True)
if __name__=='__main__':main()
