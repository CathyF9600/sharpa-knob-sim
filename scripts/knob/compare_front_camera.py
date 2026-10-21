"""Replay original Dex-X optical calibration versus frontal and oblique alternatives."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from cpu_rollouts import PreviewSim,KnobGeometry,p,font
from visual_markings import Markings
from dexx.deploy_config import ARM_BASE_POS,SIM_INTRINSICS
root=Path(__file__).resolve().parents[2];source=root/'logs/knob180_pinch_4x50/preview';out=root/'logs/knob180_front_camera';out.mkdir(exist_ok=True)
pose=json.loads((source/'cpu_pregrasp.json').read_text());g=KnobGeometry(**json.loads((source/'geometry.json').read_text()));center=np.array(pose['fixture_position_m'])+[0,0,g.rotor_center_z_m];target=center+[0,0,.025]
T=np.load(root/'calib/camera_align/current.npy');eye=T[:3,3]+ARM_BASE_POS
cameras=[{'name':'Dex-X shipped calibration','eye':eye,'target':eye+T[:3,2],'up':-T[:3,1]},
 {'name':'Front closer: 25 deg down','eye':target+[.5,0,.5*np.tan(np.deg2rad(25))],'target':target,'up':[0,0,1]},
 {'name':'Previous oblique: 35 deg down','eye':np.array([.01142465,.16857536,.80946696]),'target':target,'up':[0,0,1]}]
w,h=320,240;fx,fy,cx,cy=[SIM_INTRINSICS[k] for k in ['fx','fy','cx','cy']];near=.01
proj=p.computeProjectionMatrix(-cx/fx*near,(w-cx)/fx*near,-(h-cy)/fy*near,cy/fy*near,near,3)
for cam in cameras:
 cam['view']=p.computeViewMatrix(cam['eye'],cam['target'],cam['up'])
 V=np.array(cam['view']).reshape(4,4,order='F');P=np.array(proj).reshape(4,4,order='F');q=P@V@np.r_[center,1];ndc=q[:3]/q[3];cam['center_pixel']=[float((ndc[0]+1)*w/2),float((1-ndc[1])*h/2)]
records=[];sheet=Image.new('RGB',(1720,3*520),'#101827');draw=ImageDraw.Draw(sheet)
p.connect(p.DIRECT)
def render(sim,marks,state,cam):
 for joint,q in zip(sim.ids,state['q_rad']):p.resetJointState(sim.robot,joint,q)
 p.resetJointState(sim.knob,0,state['angle_rad']);marks.update(state['angle_rad'])
 result=p.getCameraImage(w,h,cam['view'],proj,renderer=p.ER_TINY_RENDERER,shadow=0)
 rgb=Image.fromarray(np.asarray(result[2],dtype=np.uint8)[:,:,:3]);seg=np.asarray(result[4])&((1<<24)-1)
 return rgb,[int((seg==body).sum()) for body in marks.bodies]
for condition in range(4):
 data=[json.loads(s) for s in (source/f'{condition+1:02}.jsonl').open()];header,rows=data[0],data[1:];states=[rows[0]['before']]+[r['after'] for r in rows]
 sim=PreviewSim((source/'assets/preview_robot.urdf',source/'assets/preview_knob.urdf'),pose,g,header['brake_nm'],'solver_friction');p.changeVisualShape(sim.knob,0,rgbaColor=[0,0,0,0]);marks=Markings(root/'assets/tasks/knob/markings',center)
 for ci,cam in enumerate(cameras):
  counts=[]
  for degree in [0,30,60,90,120,150,180]:
   state=min(states,key=lambda s:abs(np.rad2deg(s['angle_rad']-header['reference_angle_rad'])-degree));im,visible=render(sim,marks,state,cam);counts.append(visible)
   if condition==3 and degree in [0,120,180]:
    row=ci*520;column=[0,120,180].index(degree)
    if degree==0:
     sheet.paste(im.resize((640,480),Image.NEAREST),(0,row+40));draw.text((10,row+8),cam['name']+' | full native 320x240, shown 2x',font=font(18),fill='white')
    x,y=map(round,cam['center_pixel']);crop=im.crop((x-60,y-60,x+60,y+60));sheet.paste(crop.resize((360,360),Image.NEAREST),(640+column*360,row+60));draw.text((650+column*360,row+10),f'{degree} deg | 120px crop, shown 3x',font=font(17),fill='white')
  records.append({'condition_id':condition,'camera':cam['name'],'eye_world_m':list(map(float,cam['eye'])),'top_side_visible_pixels_at_0_30_60_90_120_150_180':counts})
 # Animated same measured high-resistance episode, camera geometry unchanged.
 if condition==3:
  frames=[];times=np.array([0]+[r['time_s'] for r in rows])
  for t in np.arange(0,times[-1]+.5,.5):
   state=states[min(np.searchsorted(times,t,side='right')-1,len(states)-1)];canvas=Image.new('RGB',(1280,3*300),'#101827');d=ImageDraw.Draw(canvas)
   for ci,cam in enumerate(cameras):
    im,_=render(sim,marks,state,cam);y=ci*300;canvas.paste(im.resize((320,240)),(0,y+40));x0,y0=map(round,cam['center_pixel'])
    canvas.paste(im.crop((x0-60,y0-60,x0+60,y0+60)).resize((240,240),Image.NEAREST),(360,y+40))
    d.text((10,y+8),cam['name'],font=font(18),fill='white');d.text((650,y+75),f't={min(t,times[-1]):.1f}s | angle={np.rad2deg(state["angle_rad"]-header["reference_angle_rad"]):.1f} deg',font=font(21),fill='white');d.text((650,y+115),'Native frame + 2x pixel crop',font=font(19),fill='#b8c4da');d.text((650,y+150),'Full robot; measured replay; 4x speed',font=font(19),fill='#b8c4da')
   frames.append(canvas)
  durations=[125]*len(frames);durations[-1]+=1500
  frames[0].save(out/'front_vs_oblique.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0)
sheet.save(out/'front_camera_comparison.png')
(out/'comparison.json').write_text(json.dumps({'intrinsics':SIM_INTRINSICS,'native_resolution':[320,240],'calibration_source':'calib/camera_align/current.npy','scope':'4 representative episodes, 7 angles per episode; RGB geometry replay, not RealSense depth/noise simulation','results':records},indent=2)+'\n')
p.disconnect();print(out,flush=True)
