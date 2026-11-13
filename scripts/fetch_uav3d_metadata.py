#!/usr/bin/env python3
"""Bounded public metadata extraction; refuses servers that ignore HTTP Range."""
import io,json,urllib.request,zipfile
from pathlib import Path

class Remote(io.RawIOBase):
 def __init__(self,url,size,budget=30000000):
  self.url,self.size,self.budget=url,size,budget;self.pos=0;self.fetched=0;self.cache={};self.requests=[]
 def seekable(self): return True
 def readable(self): return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=(offset if whence==0 else self.pos+offset if whence==1 else self.size+offset)
  if self.pos<0:raise ValueError('negative position')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  start=self.pos;end=start+n
  for (a,b),content in self.cache.items():
   if a<=start and end<=b:self.pos=end;return content[start-a:end-a]
  if n<65536:
   a=(start//65536)*65536;b=min(self.size,max(a+65536,end))
  else:a,b=start,end
  if self.fetched+b-a>self.budget:raise RuntimeError('Configured network byte cap would be exceeded')
  req=urllib.request.Request(self.url,headers={'Range':'bytes=%d-%d'%(a,b-1),'User-Agent':'UAV-Abnormality-Detect/0.1'})
  with urllib.request.urlopen(req,timeout=60) as response:
   if response.status!=206 or response.headers.get('Content-Range')!='bytes %d-%d/%d'%(a,b-1,self.size):raise ValueError('Server ignored or changed HTTP Range; stopped')
   data=response.read(b-a+1)
   if len(data)!=b-a:raise ValueError('Unexpected range length')
  self.fetched+=len(data);self.requests.append({'start':a,'end':b-1,'size':len(data)})
  self.cache[(a,b)]=data
  if len(self.cache)>16:del self.cache[next(iter(self.cache))]
  self.pos=end
  return data[start-a:end-a]

def main():
 import argparse,hashlib,zlib
 parser=argparse.ArgumentParser(description="Fetch only six official UAV3D metadata tables via bounded ZIP HTTP ranges; no images or annotations")
 parser.add_argument('--manifest',default='configs/datasets/uav3d_mini_manifest.json')
 parser.add_argument('--destination',default='data/raw/uav3d_metadata')
 parser.add_argument('--max-download-bytes',type=int,default=30000000)
 arguments=parser.parse_args()
 manifest=json.loads(Path(arguments.manifest).read_text())
 archive=manifest['files'][0]
 remote=Remote(archive['url'],archive['size'],arguments.max_download_bytes)
 destination=Path(arguments.destination)
 names={'scene.json','sample.json','sample_data.json','ego_pose.json','calibrated_sensor.json','sensor.json'}
 receipt=[]
 with zipfile.ZipFile(remote) as handle:
  selected=[entry for entry in handle.infolist() if Path(entry.filename).name in names]
  if len(selected)!=len(names):raise ValueError('Expected exactly six official metadata tables')
  if sum(entry.compress_size for entry in selected)+remote.fetched>arguments.max_download_bytes:
   raise ValueError('Metadata exceeds network cap before extraction')
  for info in selected:
   path=destination/info.filename
   if destination.resolve() not in path.resolve().parents:raise ValueError('Unsafe archive path')
   data=path.read_bytes() if path.is_file() else None
   if data is None or len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
    data=handle.read(info) # ZIP library verifies member CRC32.
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   receipt.append({'path':info.filename,'size':len(data),'compressed_size':info.compress_size,'crc32':info.CRC,'sha256':hashlib.sha256(data).hexdigest()})
   print(info.filename,len(data),flush=True)
 result={'archive_url':archive['url'],'archive_size':archive['size'],'download_kind':'partial_metadata_only','images_downloaded':False,'annotations_downloaded':False,'fetched_bytes_this_run':remote.fetched,'files':receipt,'http_ranges':remote.requests}
 destination.mkdir(parents=True,exist_ok=True)
 (destination/'metadata_acquisition_receipt.json').write_text(json.dumps(result,indent=2))
 print('Transferred bytes:',remote.fetched)

if __name__=='__main__':
 main()
