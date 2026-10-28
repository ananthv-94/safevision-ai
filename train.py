"""Legitimate YOLO11 Nano fine-tuning on CPU. Test images never enter training."""
import argparse,hashlib,json,os,time,shutil
from pathlib import Path
import safevision
from safevision.workflow import ROOT,DATA,runtime_yaml,validate_dataset

def train(epochs=10,imgsz=320,batch=4,threads=2,output='training/cpu_person'):
    import torch
    from ultralytics import YOLO,settings
    settings.update({'mlflow':False,'wandb':False})
    output=(ROOT/output).resolve()
    if (output/'weights/best.pt').exists():
        raise ValueError('Output already contains a trained model. Choose a new --output.')
    quality=json.loads((DATA/'quality_report.json').read_text())
    checked=validate_dataset()
    config=dict(model='yolo11n.pt',model_architecture='YOLO11 Nano',device='cpu',epochs=epochs,
                imgsz=imgsz,batch=batch,threads=threads,lr0=.001,optimizer='AdamW',
                freeze=10,patience=4,seed=42,workers=0,amp=False,
                dataset_version=quality['dataset_version'],split_sizes=checked['splits'])
    model=YOLO(str(ROOT/'yolo11n.pt'))
    model.add_callback('on_pretrain_routine_start',lambda trainer:torch.set_num_threads(threads))
    start=time.perf_counter()
    model.train(data=str(runtime_yaml()),device='cpu',epochs=epochs,imgsz=imgsz,batch=batch,
                workers=0,amp=False,optimizer='AdamW',lr0=.001,lrf=.1,freeze=10,
                patience=4,seed=42,deterministic=True,cache=False,plots=True,
                mosaic=.2,close_mosaic=2,mixup=0,fliplr=.5,degrees=3,scale=.2,translate=.05,
                warmup_epochs=1,project=str(output.parent),name=output.name,exist_ok=True)
    best=output/'weights/best.pt'
    if not best.is_file(): raise RuntimeError('Training did not produce best.pt')
    original=hashlib.sha256((ROOT/'yolo11n.pt').read_bytes()).hexdigest()
    trained=hashlib.sha256(best.read_bytes()).hexdigest()
    assert trained!=original,'A copied pretrained file is not a training result'
    (ROOT/'models').mkdir(exist_ok=True)
    shutil.copyfile(best,ROOT/'models/best.pt')
    (output/'training_config.json').write_text(json.dumps(config,indent=2))
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=pd.read_csv(output/'results.csv');history.columns=history.columns.str.strip()
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,loss in zip(axes,['box_loss','cls_loss','dfl_loss']):
        for phase in ('train','val'):
            key=phase+'/'+loss
            if key in history:ax.plot(history['epoch'],history[key],label=phase)
        ax.set_title(loss);ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(output/'loss_curves.png',dpi=140);plt.close(fig)
    summary=dict(status='completed',epochs_completed=len(history),seconds=time.perf_counter()-start,
                 original_sha256=original,best_sha256=trained,best_model='models/best.pt',
                 checkpoint_selection='Highest Ultralytics validation fitness; test set not used',
                 dataset_validation=checked,configuration=config)
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2))
    from safevision.training_log import log_training
    summary['mlflow']=log_training(output,config)
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--imgsz',type=int,default=320);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--output',default='training/cpu_person')
    a=p.parse_args()
    if min(a.epochs,a.imgsz,a.batch,a.threads)<=0:p.error('Arguments must be positive')
    train(a.epochs,a.imgsz,a.batch,a.threads,a.output)
