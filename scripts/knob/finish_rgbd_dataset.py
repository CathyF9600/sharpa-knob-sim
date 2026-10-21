"""Verify complete 4x50 visual export, record grouped splits and final provenance."""
import json,sys,subprocess,hashlib,shutil
from pathlib import Path
import h5py
import numpy as np
from export_rgbd import sha,ROOT
folder=Path(sys.argv[1]).resolve()
subprocess.run([sys.executable,str(ROOT/'scripts/knob/summarize_batch180.py'),str(folder),'--goal-degrees','90'],check=True)
manifest=json.loads((folder/'manifest.json').read_text());camera=json.loads((folder/'camera/camera.json').read_text());reports=[];splits={'train':[],'validation':[],'test':[]};total_frames=0
for entry in manifest:
 name=Path(entry['accepted_file']).stem;path=folder/'episodes'/(name+'.h5');report=json.loads(path.with_suffix('.audit.json').read_text());assert sha(path)==report['h5_sha256'];assert sha(folder/entry['accepted_file'])==report['source_sha256']==entry['sha256']
 with h5py.File(path,'r') as f:
  assert f.attrs['complete'];assert json.loads(f.attrs['camera_json'])==camera
  head=json.loads(f.attrs['header_json']);assert head['goal_angle_deg']==90
  n=f['actions/joint_position_target_rad'].shape[0];assert f['observations/rgb'].shape==(n+1,240,320,3);assert f['observations/depth_m'].shape==(n+1,240,320)
  t=f['observations/timestamp_s'][:];assert np.allclose(t,np.arange(n+1)/30,atol=1e-12);assert f['transition_terminal'][:].sum()==1 and f['transition_terminal'][-1]
  assert f['observations/rgb'].dtype==np.uint8 and f['observations/depth_m'].dtype==np.float32
  assert all(report[k] for k in ['all_frames_finite_depth','alignment_verified','replay_readback_verified'])
  assert f['frame_rgb_depth_sha256'].shape==(n+1,32)
  split='test' if head['seed']%10==0 else 'validation' if head['seed']%10==1 else 'train';splits[split].append({'episode':name,'file':report['file'],'seed':head['seed'],'condition_id':head['condition_id']});total_frames+=n+1
 reports.append(report)
assert len(reports)==200 and all(sum(r['condition_id']==c for r in reports)==50 for c in range(4))
seedsets={k:{r['seed'] for r in v} for k,v in splits.items()};assert not(seedsets['train']&seedsets['validation'] or seedsets['train']&seedsets['test'] or seedsets['test']&seedsets['validation'])
(folder/'splits.json').write_text(json.dumps({'rule':'seed % 10: 0=test, 1=validation, others=train; matched seeds never cross splits','splits':splits},indent=2)+'\n')
(folder/'visual_manifest.json').write_text(json.dumps(reports,indent=2)+'\n')
verification={'complete':True,'episodes':200,'counts_by_condition':{str(c):50 for c in range(4)},'goal_degrees':90,'total_observation_frames':total_frames,'total_transitions':total_frames-200,'image_shape':[240,320,3],'fps':30,'source_extrinsic_sha256':camera['source_extrinsic_sha256'],'all_episode_file_hashes_verified':True,'all_source_hashes_verified':True,'three_frame_replay_check_per_episode':True,'all_frames_generated_with_finite_metric_depth':True,'action_observation_alignment_verified':True,'grouped_seed_splits_verified':True,'split_counts':{k:len(v) for k,v in splits.items()},'total_h5_bytes':sum(r['bytes'] for r in reports),'max_table_depth_99percentile_error_m':max(r['table_depth_99percentile_error_max_m'] for r in reports),'sensor_noise_simulated':False,'isaac_validated':False,'optimality_claim':False}
(folder/'visual_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
source=folder/'source/visual_export';source.mkdir(exist_ok=True)
for name in ['export_rgbd.py','visual_markings.py','check_rgbd_geometry.py','preview_rgbd.py','stream_rgbd_export.py','finish_rgbd_dataset.py','summarize_batch180.py']:
 shutil.copy2(ROOT/'scripts/knob'/name,source/name)
subprocess.run([sys.executable,str(ROOT/'scripts/knob/preview_rgbd.py'),str(folder)],check=True)
readme=folder/'README.md';text=readme.read_text();text=text.replace('状态：生成中。完成标准为 `visual_verification.json` 中 `complete=true` 且 `episodes=200`。',f'状态：已完成四档各 50 条，共 200 条。共 {total_frames:,} 帧同步 RGB-D、{total_frames-200:,} 个动作转移；逐条验证结果见 `visual_verification.json`。')
readme.write_text(text)
(folder/'DELIVERY_COMPLETE.json').write_text(json.dumps(verification,indent=2)+'\n');print('DELIVERY COMPLETE',json.dumps(verification),flush=True)
