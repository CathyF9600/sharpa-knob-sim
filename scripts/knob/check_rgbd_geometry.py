"""Independent slanted-plane check of exported depth and optical pixel rays."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from export_rgbd import p,matrices,backproject
folder=Path(sys.argv[1]);cfg=json.loads((folder/'camera/camera.json').read_text());T=np.array(cfg['T_world_camera_optical']);view,proj=matrices(cfg);p.connect(p.DIRECT);reports=[]
for angles in [[0,0,0],[15,20,0],[-20,-15,20]]:
 p.resetSimulation();R=T[:3,:3]@Rotation.from_euler('xyz',angles,degrees=True).as_matrix();center=T[:3,3]+T[:3,2]*.9;n=R[:,2];front=center-n*.001
 vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[.22,.18,.001],rgbaColor=[.5,.5,.5,1]);obj=p.createMultiBody(0,-1,vis,center,Rotation.from_matrix(R).as_quat())
 result=p.getCameraImage(cfg['width'],cfg['height'],view,proj,renderer=p.ER_TINY_RENDERER);buf=np.array(result[3]);near,far=cfg['near_m'],cfg['far_m'];depth=far*near/(far-(far-near)*buf);world=backproject(depth,cfg);mask=(np.array(result[4])&((1<<24)-1))==obj
 interior=mask.copy();interior[1:-1,1:-1]&=mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:]
 error=np.abs((world[interior]-front)@n);q99=float(np.quantile(error,.99));assert q99<.0002,(angles,q99)
 reports.append({'plane_rotation_optical_deg':angles,'pixels':int(interior.sum()),'distance_error_99percentile_m':q99})
p.disconnect();(folder/'camera/ray_geometry_validation.json').write_text(json.dumps({'pass':True,'projection_raster_compensation_px':{'cx':-.5,'cy':.5},'exported_intrinsics_unchanged':True,'pixel_center':'u+0.5,v+0.5','planes':reports},indent=2)+'\n');print(reports)
