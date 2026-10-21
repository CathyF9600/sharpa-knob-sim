"""Compare visible scale area and camera-body clearance at sampled measured states."""
import json
from pathlib import Path
import numpy as np
from cpu_rollouts import PreviewSim,KnobGeometry,p
from visual_markings import Markings
root=Path(__file__).resolve().parents[2];f=root/'logs/knob180_pinch_4x50/preview';out=root/'logs/knob180_marked'
pose=json.loads((f/'cpu_pregrasp.json').read_text());g=KnobGeometry(**json.loads((f/'geometry.json').read_text()));center=np.array(pose['fixture_position_m'])+[0,0,g.rotor_center_z_m];target=center+[0,0,.025]
records=[];p.connect(p.DIRECT)
for cond in range(1,5):
 data=[json.loads(s) for s in (f/f'{cond:02}.jsonl').open()];h,rows=data[0],data[1:];states=[rows[0]['before']]+[r['after'] for r in rows]
 sim=PreviewSim((f/'assets/preview_robot.urdf',f/'assets/preview_knob.urdf'),pose,g,h['brake_nm'],'solver_friction');p.changeVisualShape(sim.knob,0,rgbaColor=[0,0,0,0]);marks=Markings(root/'assets/tasks/knob/markings',center)
 sphere=p.createCollisionShape(p.GEOM_SPHERE,radius=.04);cam=p.createMultiBody(0,sphere,-1,[0,0,2])
 for yaw in [-135,-25,45,135]:
  view=p.computeViewMatrixFromYawPitchRoll(target,.55,yaw,-35,0,2);eye=np.linalg.inv(np.array(view).reshape(4,4,order='F'))[:3,3]
  p.resetBasePositionAndOrientation(cam,eye,[0,0,0,1]);counts=[];clearance=.3
  for degree in [0,30,60,90,120,150,180]:
   state=min(states,key=lambda s:abs(np.rad2deg(s['angle_rad']-h['reference_angle_rad'])-degree))
   for joint,q in zip(sim.ids,state['q_rad']):p.resetJointState(sim.robot,joint,q)
   p.resetJointState(sim.knob,0,state['angle_rad']);marks.update(state['angle_rad'])
   seg=np.asarray(p.getCameraImage(256,256,view,p.computeProjectionMatrixFOV(30,1,.01,3),renderer=p.ER_TINY_RENDERER,shadow=0)[4])&((1<<24)-1)
   counts.append(int(np.isin(seg,marks.bodies).sum()))
   contacts=p.getClosestPoints(cam,sim.robot,.3)
   if contacts:clearance=min(clearance,min(c[8] for c in contacts))
  records.append({'condition':cond-1,'yaw_deg':yaw,'elevation_deg':35,'distance_m':.55,'fov_y_deg':30,'eye_world_m':eye.tolist(),'target_world_m':target.tolist(),'visible_scale_pixels_at_0_30_60_90_120_150_180':counts,'minimum_sampled_clearance_of_4cm_camera_sphere_m':clearance})
(out/'camera_visibility_check.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2));p.disconnect()
