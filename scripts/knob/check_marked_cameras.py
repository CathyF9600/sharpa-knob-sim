"""Replay measured poses from full robot to assess camera occlusion (no physics step)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from cpu_rollouts import PreviewSim,KnobGeometry,p,font
from visual_markings import build,Markings
ROOT=Path(__file__).resolve().parents[2]
f=ROOT/'logs/knob180_pinch_4x50/preview';out=ROOT/'logs/knob180_marked';out.mkdir(exist_ok=True)
pose=json.loads((f/'cpu_pregrasp.json').read_text());g=KnobGeometry(**json.loads((f/'geometry.json').read_text()))
data=[json.loads(s) for s in (f/'04.jsonl').open()];header,rows=data[0],data[1:]
center=np.array(pose['fixture_position_m'])+[0,0,g.rotor_center_z_m]
markfolder=build(ROOT/'assets/tasks/knob/markings');p.connect(p.DIRECT)
sim=PreviewSim((f/'assets/preview_robot.urdf',f/'assets/preview_knob.urdf'),pose,g,.1,'solver_friction')
p.changeVisualShape(sim.knob,0,rgbaColor=[0,0,0,0])
mark=Markings(markfolder,center)
# The existing yellow zero stripe is an underlying visual only.
views=[(-25,-45),(45,-45),(135,-45),(-135,-45)]
canvas=Image.new('RGB',(1440,4*390),'#101827');draw=ImageDraw.Draw(canvas);cams=[]
for ci,(yaw,pitch) in enumerate(views):
 target=center+np.array([0,0,.025]);distance=.40
 view=p.computeViewMatrixFromYawPitchRoll(target,distance,yaw,pitch,0,2)
 inv=np.linalg.inv(np.array(view).reshape(4,4,order='F'));eye=inv[:3,3]
 cams.append({'yaw_deg':yaw,'pitch_deg':pitch,'distance_m':distance,'eye_world_m':eye.tolist(),'target_world_m':target.tolist(),'up_world':[0,0,1],'fov_y_deg':45})
 for j,angle in enumerate([0,90,180]):
  state=min([rows[0]['before']]+[r['after'] for r in rows],key=lambda s:abs(np.rad2deg(s['angle_rad']-header['reference_angle_rad'])-angle))
  for joint,q in zip(sim.ids,state['q_rad']):p.resetJointState(sim.robot,joint,q)
  p.resetJointState(sim.knob,0,state['angle_rad']);mark.update(state['angle_rad'])
  rgba=np.array(p.getCameraImage(480,350,view,p.computeProjectionMatrixFOV(45,480/350,.01,3),renderer=p.ER_TINY_RENDERER,shadow=0)[2],dtype=np.uint8)
  canvas.paste(Image.fromarray(rgba[:,:,:3]),(480*j,ci*390+40))
  draw.text((480*j+10,ci*390+8),f'yaw {yaw} | elevation 45 | measured ~{angle} deg',font=font(17),fill='white')
canvas.save(out/'camera_comparison.png');(out/'camera_candidates.json').write_text(json.dumps(cams,indent=2)+'\n')
# Clear view of bare knob for checking both scales.
for name,link in sim.links.items():p.changeVisualShape(sim.robot,link,rgbaColor=[0,0,0,0])
p.resetJointState(sim.knob,0,0);mark.update(0)
view=p.computeViewMatrixFromYawPitchRoll(center,.14,45,-40,0,2)
rgba=np.array(p.getCameraImage(900,900,view,p.computeProjectionMatrixFOV(45,1,.01,3),renderer=p.ER_TINY_RENDERER,shadow=0)[2],dtype=np.uint8)
Image.fromarray(rgba[:,:,:3]).save(out/'marked_knob_detail.png');p.disconnect()
