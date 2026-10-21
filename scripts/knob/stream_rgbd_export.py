"""Export deterministic completed prefixes while physics collection continues."""
import argparse,json,time,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from export_rgbd import export_one,camera_setup,build,sha,ROOT

def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args();folder=args.folder.resolve();camera_setup(folder);build(folder/'assets/markings');(folder/'episodes').mkdir(exist_ok=True)
 scheduled=set();results={};pending={};cached={};start=time.time()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  while time.time()-start<8*3600:
   audits=[]
   for path in (folder/'attempts').glob('*.audit.json'):
    try:audits.append(json.loads(path.read_text()))
    except ValueError:pass
   ready=[]
   for c in range(4):
    rows={r['trial_id']:r for r in audits if r['condition_id']==c};end=0
    while end in rows:end+=1
    good=[rows[i] for i in range(end) if rows[i]['quality_gate_pass']][:50]
    for index,r in enumerate(good):
     name=f'c{c}_{index:03}';source=folder/r['file'];dest=folder/'accepted'/(name+'.jsonl')
     if dest.exists():assert os.path.samefile(source,dest),'Accepted deterministic prefix changed'
     else:os.link(source,dest)
     if name not in cached:cached[name]={**r,'accepted_file':str(dest.relative_to(folder)),'sha256':sha(source)}
     ready.append((name,cached[name]))
   for name,entry in ready:
    if name not in scheduled and len(pending)<args.workers*2:
     pending[pool.submit(export_one,(str(folder),entry))]=name;scheduled.add(name)
   for future in list(pending):
    if future.done():
     name=pending.pop(future);results[name]=future.result();print('EXPORTED',name,len(results),flush=True)
   atomic(folder/'visual_progress.json',sorted(results.values(),key=lambda r:r['episode']))
   atomic(folder/'pipeline_status.json',{'phase':'rendering' if pending else 'waiting_for_physics','quality_counts':{str(c):min(50,sum(r['condition_id']==c and r['quality_gate_pass'] for r in audits)) for c in range(4)},'stable_ready':len(ready),'visual_complete':len(results),'visual_pending':len(pending)})
   if len(results)==200:
    try:manifest=json.loads((folder/'manifest.json').read_text())
    except (OSError,ValueError):manifest=[]
    if len(manifest)==200:break
   time.sleep(5)
  else:raise RuntimeError('Export timed out before 200 episodes')
 atomic(folder/'pipeline_status.json',{'phase':'final_verification','visual_complete':200})
 subprocess.run([sys.executable,str(ROOT/'scripts/knob/finish_rgbd_dataset.py'),str(folder)],check=True)
 atomic(folder/'pipeline_status.json',{'phase':'complete','visual_complete':200})
if __name__=='__main__':main()
