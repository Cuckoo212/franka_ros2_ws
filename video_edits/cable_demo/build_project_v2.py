from pathlib import Path
from PIL import Image,ImageDraw
import xml.etree.ElementTree as E
import uuid,json
root=Path('/home/flexcycle/franka_ros2_ws/video_edits/cable_demo')
source='/home/flexcycle/Videos/Screencasts/demo1_10-26-2026 06:30:41 PM.webm'
fps=30; length=6716; uid='{'+str(uuid.uuid4())+'}'
events=[(988,1278,281,8),(2478,1250,344,8),(3988,1238,443,7),(5362,1244,684,7)]
mlt=E.Element('mlt',{'producer':'main_bin','version':'7.32.0','root':str(root)})
E.SubElement(mlt,'profile',{'description':'HD 1080p30','width':'1920','height':'1080','frame_rate_num':'30','frame_rate_den':'1','progressive':'1','sample_aspect_num':'1','sample_aspect_den':'1','display_aspect_num':'16','display_aspect_den':'9','colorspace':'709'})
def props(el,values):
 for k,v in values.items(): E.SubElement(el,'property',name=k).text=str(v)
def producer(id,path,service,kid,name,duration=length):
 p=E.SubElement(mlt,'producer',id=id,**{'in':'0','out':str(duration-1)})
 props(p,{'resource':path,'mlt_service':service,'length':duration,'eof':'pause','kdenlive:id':kid,'kdenlive:clipname':name,'kdenlive:duration':duration,'set.test_audio':1})
 return p
producer('black','0','color',0,'Background')
producer('video',source,'avformat',4,'Original demo')
for i,(end,x,y,r) in enumerate(events,1):
 im=Image.new('RGBA',(1920*3,1080*3));d=ImageDraw.Draw(im)
 def scale(pts):return [tuple(v*3 for v in p) for p in pts]
 d.ellipse(tuple(v*3 for v in (x-r,y-r,x+r,y+r)),outline=(255,30,30,255),width=12)
 # Arrow approaches from clear space above-right and stops at the ring.
 start=(x+75,y-65);tip=(x+r+4,y-r-4)
 d.line(scale([start,tip]),fill=(255,30,30,255),width=12)
 import math
 dx=start[0]-tip[0];dy=start[1]-tip[1];l=math.hypot(dx,dy);ux=dx/l;uy=dy/l
 d.polygon(scale([tip,(tip[0]+14*ux+6*uy,tip[1]+14*uy-6*ux),(tip[0]+14*ux-6*uy,tip[1]+14*uy+6*ux)]),fill=(255,30,30,255))
 im=im.resize((1920,1080),Image.LANCZOS);p=root/'assets'/f'selection_v2_{i}.png';im.save(p)
 producer('overlay'+str(i),str(p),'qimage',4+i,f'Selection {i}: circle and arrow',150)
black=E.SubElement(mlt,'playlist',id='black_track');E.SubElement(black,'entry',producer='black',**{'in':'0','out':str(length-1)})
for track in (0,1):
 pl=E.SubElement(mlt,'playlist',id=f'playlist{track*2}')
 if track==0:E.SubElement(pl,'entry',producer='video',**{'in':'0','out':str(length-1)})
 else:
  cursor=0
  for i,(end,*_) in enumerate(events,1):
   start=end-150
   if start>cursor:E.SubElement(pl,'blank',length=str(start-cursor))
   E.SubElement(pl,'entry',producer='overlay'+str(i),**{'in':'0','out':'149'});cursor=end
 E.SubElement(mlt,'playlist',id=f'playlist{track*2+1}')
 tr=E.SubElement(mlt,'tractor',id=f'track{track}',**{'in':'0','out':str(length-1)})
 props(tr,{'kdenlive:track_name':['Demo video','Selection annotations'][track],'kdenlive:trackheight':80,'kdenlive:timeline_active':1})
 for j in (0,1):E.SubElement(tr,'track',producer=f'playlist{track*2+j}',hide='audio')
seq=E.SubElement(mlt,'tractor',id='sequence',**{'in':'0','out':str(length-1)})
props(seq,{'kdenlive:uuid':uid,'kdenlive:id':3,'kdenlive:clipname':'Cable selection demo','kdenlive:sequenceproperties.activeTrack':1,'kdenlive:sequenceproperties.tracks':2,'kdenlive:sequenceproperties.hasAudio':0,'kdenlive:sequenceproperties.hasVideo':1,'kdenlive:sequenceproperties.duration':length,'kdenlive:sequenceproperties.position':2350,'kdenlive:sequenceproperties.zonein':0,'kdenlive:sequenceproperties.zoneout':length,'kdenlive:sequenceproperties.guides':json.dumps([{'pos':e/30,'comment':f'Selection {i}: TF appears','type':0} for i,(e,*_) in enumerate(events,1)])})
for p in ['black_track','track0','track1']:E.SubElement(seq,'track',producer=p)
t=E.SubElement(seq,'transition',**{'in':'0','out':str(length-1)})
props(t,{'mlt_service':'frei0r.cairoblend','a_track':1,'b_track':2,'disable':0,'always_active':1,'internal_added':237})
bin=E.SubElement(mlt,'playlist',id='main_bin')
props(bin,{'kdenlive:docproperties.version':'1.1','kdenlive:docproperties.kdenliveversion':'26.08.2','kdenlive:docproperties.documentid':'1793190000','kdenlive:docproperties.uuid':uid,'kdenlive:docproperties.activetimeline':uid,'kdenlive:docproperties.opensequences':uid,'kdenlive:docproperties.profile':'atsc_1080p_30','kdenlive:docproperties.enableproxy':0})
for p in ['video','overlay1','overlay2','overlay3','overlay4','sequence']:E.SubElement(bin,'entry',producer=p)
last=E.SubElement(mlt,'tractor',id='project',**{'in':'0','out':str(length-1)})
props(last,{'kdenlive:projectTractor':1});E.SubElement(last,'track',producer='sequence',**{'in':'0','out':str(length-1)})
E.indent(mlt);E.ElementTree(mlt).write(root/'cable_selection_v2.kdenlive',encoding='utf-8',xml_declaration=True)
(root/'timing_v2.json').write_text(json.dumps([{'tf_time_seconds':e/30,'annotation_start_seconds':(e-150)/30,'annotation_end_seconds':e/30,'center_xy':[x,y],'diameter_px':r*2,'stroke_px':4} for e,x,y,r in events],indent=2))
print(root)
