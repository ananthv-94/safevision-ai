"""Small SH17 person-only subset. Existing human annotations, not pseudo-labels."""
import argparse, csv, hashlib, json, random, shutil, urllib.request, urllib.parse, zipfile, io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMIT='2f832c289f9deea063367045ed8230b26e3e7ea9'
URL=f'https://raw.githubusercontent.com/ahmadmughees/SH17dataset/{COMMIT}/data/list_of_all_urls.csv'
CSV_SHA='85b04a113bb833b9c4cc9203142b4c59df4e4db083da635c48159d8470fae7ef'
KAGGLE='https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection/'
LICENSE='CC BY-NC-SA 4.0; original photographs also subject to Pexels terms'
def sha(data): return hashlib.sha256(data).hexdigest()
def fetch(url,path):
    path=Path(path)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=90) as response:
            path.write_bytes(response.read())
    return path.read_bytes()
def parse_labels(text):
    boxes=[]
    for line in text.splitlines():
        values=list(map(float,line.split()))
        if not values: continue
        if len(values)!=5: raise ValueError('Invalid source annotation')
        cls,x,y,w,h=values
        if cls!=0: continue
        if not np.isfinite(values).all() or w<=0 or h<=0: raise ValueError('Invalid box')
        x1,y1,x2,y2=np.clip([x-w/2,y-h/2,x+w/2,y+h/2],0,1)
        if x2>x1 and y2>y1: boxes.append([0,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
    return boxes
def phash(image):
    small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32)
    low=cv2.dct(small)[:8,:8].flatten()[1:]
    bits=low>np.median(low)
    return sum(int(b)<<i for i,b in enumerate(bits))
def choose_subset(groups,target):
    dp={0:[]}
    for key,items in groups.items():
        for size,keys in list(dp.items()):
            new=size+len(items)
            if new<=target+8 and new not in dp: dp[new]=keys+[key]
    return dp[min(dp,key=lambda n:abs(n-target))]
def prepare(count=120,seed=42):
    out=ROOT/'datasets/safevision'
    cache=ROOT/'.cache/sh17'
    cache.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    data=fetch(URL,cache/'source_urls.csv')
    assert sha(data)==CSV_SHA,'Upstream URL manifest changed'
    photos=sorted((int(u.split('/')[-2]),u) for u in data.decode().splitlines()
                  if u.split('/')[-1].startswith('pexels-photo-'))
    groups=[]
    for pid,url in photos:
        if not groups or pid-groups[-1][-1][0]>2000: groups.append([])
        groups[-1].append((pid,url))
    groups=[g for g in groups if 3<=len(g)<=24]
    random.Random(seed).shuffle(groups)
    candidates=[(g[0][0],pid,url) for g in groups[:60] for pid,url in g[:4]]
    def labels(item):
        group,pid,url=item
        stem=Path(urllib.parse.urlparse(url).path).stem
        endpoint=KAGGLE+urllib.parse.quote('labels/'+stem+'.txt',safe='')+'?datasetVersionNumber=1'
        try:
            raw=fetch(endpoint,cache/(stem+'.txt'))
            boxes=parse_labels(raw.decode())
            return dict(group=str(group),photo_id=pid,source_url=url,label_url=endpoint,
                        source_label_sha256=sha(raw),boxes=boxes,stem=stem) if boxes else None
        except Exception as exc:
            print(f'Label unavailable {stem}: {exc}',flush=True)
            return None
    with ThreadPoolExecutor(max_workers=8) as pool: records=[r for r in pool.map(labels,candidates) if r][:count]
    if len(records)<45: raise RuntimeError('Insufficient person-labeled images downloaded')
    def _image(record):
        r=dict(record)
        try:
            raw=fetch(r['source_url']+'?w=960',cache/(r['stem']+'.source'))
        except Exception:
            endpoint=KAGGLE+urllib.parse.quote('images/'+Path(urllib.parse.urlparse(r['source_url']).path).name,safe='')+'?datasetVersionNumber=1'
            raw=fetch(endpoint,cache/(r['stem']+'.source'))
        if raw[:2]==b'PK':
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                raw=archive.read(next(n for n in archive.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))))
        frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if frame is None: raise ValueError('Corrupt source image: '+r['stem'])
        scale=min(1.0,960/max(frame.shape[:2]))
        if scale<1: frame=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
        r['image_sha256']=sha(frame.tobytes()); r['phash']=phash(frame)
        r['height'],r['width']=frame.shape[:2]
        r['frame']=frame
        return r
    failures=[]
    def image(record):
        for attempt in range(3):
            try:return _image(record)
            except Exception as exc:
                if attempt==2:
                    failures.append(dict(image=record['stem'],error=str(exc)))
                    print('Skipped unavailable image '+record['stem'],flush=True)
        return None
    with ThreadPoolExecutor(max_workers=6) as pool: records=[r for r in pool.map(image,records) if r]
    if len(records)<45: raise RuntimeError('Too few usable images')
    # Exact duplicates are removed; near duplicates merge inferred shooting groups.
    unique={}
    for r in records: unique.setdefault(r['image_sha256'],r)
    duplicates=len(records)-len(unique); records=list(unique.values())
    parent={r['group']:r['group'] for r in records}
    def find(g):
        while parent[g]!=g: g=parent[g]
        return g
    near_pairs=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if (a['phash']^b['phash']).bit_count()<=8:
                parent[find(b['group'])]=find(a['group'])
                near_pairs.append([a['stem'],b['stem']])
    grouped={}
    for r in records:
        r['group']=find(r['group'])
        grouped.setdefault(r['group'],[]).append(r)
    keys=list(grouped); random.Random(seed+1).shuffle(keys)
    grouped={key:grouped[key] for key in keys}
    splits={}
    for split in ('test','val'):
        chosen=choose_subset(grouped,round(len(records)*.15))
        splits[split]=[r for key in chosen for r in grouped.pop(key)]
    splits['train']=[r for items in grouped.values() for r in items]
    if not all(splits.values()): raise RuntimeError('Grouping left an empty split')
    manifest=[]
    # Reruns only replace this script's dataset directories, never arbitrary user paths.
    for split,items in splits.items():
        for kind in ('images','labels'):
            directory=out/kind/split
            directory.mkdir(parents=True,exist_ok=True)
            for p in directory.iterdir():
                if p.is_file(): p.unlink()
        for r in items:
            image_path=out/'images'/split/(r['stem']+'.jpg')
            assert cv2.imwrite(str(image_path),r['frame'],[cv2.IMWRITE_JPEG_QUALITY,92])
            label_path=out/'labels'/split/(r['stem']+'.txt')
            label_path.write_text('\n'.join(' '.join(str(v) if j==0 else f'{v:.6f}' for j,v in enumerate(box)) for box in r['boxes'])+'\n')
            manifest.append({k:v for k,v in r.items() if k not in ('frame','phash')}|
                            dict(split=split,image_file='images/'+split+'/'+image_path.name,image_file_sha256=sha(image_path.read_bytes())))
    (out/'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n')
    # Scripts pass an absolute, relocated YAML copy to Ultralytics; this source remains portable.
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    version=sha(json.dumps(manifest,sort_keys=True).encode())
    report=dict(dataset='SH17 person-only subset',license=LICENSE,source_commit=COMMIT,
                annotation_source='SH17 original human YOLO labels; retain class 0 only',
                unavailable_images=failures,images=len(records),instances=sum(len(r['boxes']) for r in records),
                splits={s:dict(images=len(rs),instances=sum(len(r['boxes']) for r in rs),groups=len({r['group'] for r in rs})) for s,rs in splits.items()},
                classes={'0':'person'},missing_labels=0,corrupt_images=0,
                exact_duplicates_removed=duplicates,near_duplicate_pairs=len(near_pairs),
                cross_split_group_overlap=0,cross_split_exact_duplicates=0,
                dataset_version=version,
                grouping='Conservative inferred shoots: connected source photo IDs <=2000 apart, then pHash distance <=8 merging. Not official camera/session metadata.',
                limitations='Small positive-only still-photo subset. No site/camera IDs; residual undetected scene similarity is possible. Not a production site holdout.',
                near_duplicate_pairs_detail=near_pairs)
    (out/'quality_report.json').write_text(json.dumps(report,indent=2))
    (out/'LICENSE_AND_SOURCE.md').write_text('# SH17 subset\n\nSource: https://github.com/ahmadmughees/SH17dataset\nKaggle: https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection\n\nAnnotations: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/ .\nPhotographs retain Pexels terms: https://www.pexels.com/license/ .\nNoncommercial educational/interview experiment; no endorsement or misconduct claim.\nCredit: Hafiz Mughees Ahmad and Afshin Rahimi, SH17, 2024, DOI 10.1016/j.jnlssr.2024.09.002.\n\nChanges: select a small person-positive subset; retain human class-0 annotations; clip box edges to image; resize uncropped photographs; group and split. Per-image original URL and original annotation hashes are in manifest.json. No model-generated labels.\n')
    qa=out/'visualizations';qa.mkdir(exist_ok=True)
    for page,start in enumerate(range(0,len(manifest),12)):
        canvas=np.full((720,1280,3),245,np.uint8)
        for n,r in enumerate(manifest[start:start+12]):
            frame=cv2.imread(str(out/r['image_file']))
            h,w=frame.shape[:2]
            for _,x,y,bw,bh in r['boxes']:
                cv2.rectangle(frame,(int((x-bw/2)*w),int((y-bh/2)*h)),(int((x+bw/2)*w),int((y+bh/2)*h)),(0,220,0),2)
            ratio=min(320/w,210/h); frame=cv2.resize(frame,(int(w*ratio),int(h*ratio)))
            y,x=divmod(n,4); yy,xx=y*240,x*320
            canvas[yy:yy+frame.shape[0],xx:xx+frame.shape[1]]=frame
            cv2.putText(canvas,f"{r['photo_id']} {r['split']} g{r['group']}",(xx+4,yy+232),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
        cv2.imwrite(str(qa/f'annotations_{page:02d}.jpg'),canvas)
    print(json.dumps(report,indent=2),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=120);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();prepare(a.count,a.seed)
