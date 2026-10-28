"""Dataset validation and portable paths for CPU training/evaluation."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/safevision'
def runtime_yaml(data=DATA):
    data=Path(data).resolve()
    config=yaml.safe_load((data/'data.yaml').read_text())
    config['path']=str(data)
    target=ROOT/'.cache/data_runtime.yaml'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(yaml.safe_dump(config))
    return target
def validate_dataset(data=DATA):
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text())
    seen_groups,seen_hashes={},{}
    counts={s:0 for s in ('train','val','test')}
    boxes=0
    for row in manifest:
        split=row['split']; path=data/row['image_file']
        image=cv2.imread(str(path))
        if image is None: raise ValueError('Corrupt image '+str(path))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['image_file_sha256']
        for key,seen in [(row['group'],seen_groups),(row['image_sha256'],seen_hashes)]:
            if key in seen and seen[key]!=split: raise ValueError('Cross-split leakage')
            seen[key]=split
        labels=data/'labels'/split/(path.stem+'.txt')
        values=np.loadtxt(labels,ndmin=2)
        if values.shape[1]!=5 or not np.isfinite(values).all(): raise ValueError('Bad label shape')
        if not (values[:,0]==0).all(): raise ValueError('Unexpected class')
        xy,wh=values[:,1:3],values[:,3:5]
        if not ((wh>0).all() and (xy-wh/2>=-1e-6).all() and (xy+wh/2<=1.000001).all()):
            raise ValueError('Box outside image')
        assert len(values)==len(row['boxes'])
        boxes+=len(values);counts[split]+=1
    assert all(counts.values())
    return dict(images=len(manifest),boxes=boxes,splits=counts,leakage_checks='passed')
