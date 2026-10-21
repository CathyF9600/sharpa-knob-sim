"""Visual-only rotor scales and dimensioned print templates; never collision geometry."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p

COLORS={0:'#ffcc00',90:'#ffffff',180:'#ff7043',270:'#55e0dc'}
def font(n):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',n)

def build(folder,r=.025,h=.020):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    top=Image.new('RGB',(1024,1024),'#173f58');d=ImageDraw.Draw(top)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="50mm" height="50mm" viewBox="0 0 1024 1024">','<circle cx="512" cy="512" r="512" fill="#173f58"/>']
    for deg in range(0,360,15):
        a=math.radians(deg);major=deg%30==0;color=COLORS.get(deg,'#ffffff');w=10 if major else 5
        pts=[(512+rr*math.cos(a),512-rr*math.sin(a)) for rr in (410 if major else 455,493)]
        d.line(pts,fill=color,width=w)
        svg.append(f'<line x1="{pts[0][0]}" y1="{pts[0][1]}" x2="{pts[1][0]}" y2="{pts[1][1]}" stroke="{color}" stroke-width="{w}"/>')
        if deg%90==0:
            x,y=512+330*math.cos(a),512-330*math.sin(a)
            d.text((x,y),str(deg),font=font(78),anchor='mm',fill=color)
            svg.append(f'<text x="{x}" y="{y}" dominant-baseline="central" text-anchor="middle" font-family="DejaVu Sans" font-weight="bold" font-size="78" fill="{color}">{deg}</text>')
    d.ellipse((425,425,599,599),fill='#8b9198')
    svg+=['<circle cx="512" cy="512" r="87.04" fill="white" stroke="black" stroke-dasharray="8 8"/>','</svg>']
    (folder/'top_scale_50mm.svg').write_text('\n'.join(svg));top.save(folder/'top.png')
    side=Image.new('RGB',(2048,256),'#173f58');d=ImageDraw.Draw(side)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*math.pi*r*1000:.5f}mm" height="{h*1000}mm" viewBox="0 0 2048 256">','<rect width="2048" height="256" fill="#173f58"/>']
    # Seam at 345 deg keeps zero fully visible; UV uses the same offset.
    for deg in range(0,360,15):
        x=((deg+15)%360)/360*2048;color=COLORS.get(deg,'#ffffff');major=deg%30==0
        length=58 if major else 30
        d.line((x,4,x,length),fill=color,width=7 if major else 3)
        d.line((x,252-length,x,252),fill=color,width=7 if major else 3)
        svg.append(f'<path d="M {x} 4 V {length} M {x} {252-length} V 252" stroke="{color}" stroke-width="{7 if major else 3}"/>')
        if major:
            d.text((x,128),str(deg),font=font(35),anchor='mm',fill=color)
            svg.append(f'<text x="{x}" y="128" dominant-baseline="central" text-anchor="middle" font-family="DejaVu Sans" font-weight="bold" font-size="35" fill="{color}">{deg}</text>')
    (folder/'side_scale_157mm.svg').write_text('\n'.join(svg+['</svg>']));side.save(folder/'side.png')
    # Separate textured visual meshes; 0.05 mm offset prevents z-fighting.
    for kind in ['top','side']:
        vertices=[];uv=[];faces=[];n=192
        for i in range(n+1):
            a=2*math.pi*i/n-math.pi/12
            for j in range(2):
                if kind=='top':
                    rr=[.00425,r][j];x,y=rr*math.cos(a),rr*math.sin(a)
                    vertices.append((x,y,h/2+.00005));uv.append((.5+x/(2*r),.5+y/(2*r)))
                else:
                    rr=r+.00005;vertices.append((rr*math.cos(a),rr*math.sin(a),[-h/2,h/2][j]));uv.append((i/n,j))
            if i<n:
                k=2*i
                faces.extend([(k,k+1,k+3),(k,k+3,k+2)] if kind=='top' else [(k,k+2,k+3),(k,k+3,k+1)])
        lines=[f'v {x} {y} {z}' for x,y,z in vertices]+[f'vt {u} {v}' for u,v in uv]
        lines += ['f '+' '.join(f'{i+1}/{i+1}' for i in f) for f in faces]
        (folder/f'{kind}.obj').write_text('\n'.join(lines)+'\n')
    return folder

class Markings:
    def __init__(self,folder,center,r=.025,h=.020):
        self.center=center;self.bodies=[]
        # White stationary reference arrow painted on the flange, pointing inward.
        z=-.054+.008+.003  # Lift above Bullet's rounded primitive visual cap.
        visual=p.createVisualShape(p.GEOM_MESH,vertices=[[.027,0,z],[.043,-.004,z],[.043,.004,z]],indices=[0,1,2],rgbaColor=[1,1,1,1])
        self.reference=p.createMultiBody(0,-1,visual,center)
        for kind in ['top','side']:
            visual=p.createVisualShape(p.GEOM_MESH,fileName=str(Path(folder)/f'{kind}.obj'),rgbaColor=[1,1,1,1])
            body=p.createMultiBody(0,-1,visual,center)
            p.changeVisualShape(body,-1,textureUniqueId=p.loadTexture(str(Path(folder)/f'{kind}.png')))
            self.bodies.append(body)
    def update(self,angle):
        for body in self.bodies:p.resetBasePositionAndOrientation(body,self.center,p.getQuaternionFromEuler([0,0,angle]))
