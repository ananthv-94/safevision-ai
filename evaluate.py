"""Official Ultralytics validation, or a fixed held-out pretrained/fine-tuned comparison."""
import argparse,json
from pathlib import Path
import safevision
from safevision.workflow import ROOT,DATA,runtime_yaml,validate_dataset

def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union else 0.

def fixed_iou_metrics(model,imgsz=320,threads=2):
    import numpy as np,cv2
    from safevision.engines import TorchEngine,box_iou
    engine=TorchEngine(model,threads,imgsz)
    tp=fp=fn=0
    matched_iou=[]
    for image_path in sorted((DATA/'images/test').glob('*.jpg')):
        image=cv2.imread(str(image_path));h,w=image.shape[:2]
        labels=np.loadtxt(DATA/'labels/test'/(image_path.stem+'.txt'),ndmin=2)
        xy=labels[:,1:3]*[w,h];wh=labels[:,3:5]*[w,h]
        truth=np.c_[xy-wh/2,xy+wh/2]
        predictions,_=engine.detect(image,confidence=.25,iou_threshold=.7)
        used=set()
        for box in predictions:
            candidates=[i for i in range(len(truth)) if i not in used]
            if candidates:
                overlaps=box_iou(box[:4],truth[candidates]);index=int(np.argmax(overlaps))
                if overlaps[index]>=.5:
                    used.add(candidates[index]);tp+=1;matched_iou.append(float(overlaps[index]));continue
            fp+=1
        fn+=len(truth)-len(used)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return dict(confidence=.25,matching_iou=.5,nms_iou=.7,tp=tp,fp=fp,fn=fn,precision=p,recall=r,
                f1=2*p*r/(p+r) if p+r else 0,mean_matched_iou=float(np.mean(matched_iou)) if matched_iou else None,
                preprocessing='Square letterbox, same as engine benchmark; separate fixed-threshold analysis')

def evaluate(model,data,split='test',output='reports/evaluation',imgsz=320,threads=2):
    import torch,numpy as np
    from ultralytics import YOLO
    detector=YOLO(str(model))
    detector.add_callback('on_val_start',lambda validator:torch.set_num_threads(threads))
    output=Path(output).resolve()
    result=detector.val(data=str(data),split=split,classes=[0],device='cpu',imgsz=imgsz,batch=1,
                        workers=0,half=False,conf=.001,iou=.7,max_det=300,plots=True,
                        project=str(output.parent),name=output.name,exist_ok=True,verbose=False)
    precision,recall=float(result.box.mp),float(result.box.mr)
    metrics=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                 map50=float(result.box.map50),map50_95=float(result.box.map),speed_ms=result.speed)
    metrics['per_class']=[]
    for index,class_id in enumerate(result.box.ap_class_index):
        metrics['per_class'].append(dict(class_id=int(class_id),name=detector.names[int(class_id)],
              precision=float(result.box.p[index]),recall=float(result.box.r[index]),
              ap50=float(result.box.ap50[index]),ap50_95=float(result.box.ap[index])))
    metrics['settings']=dict(split=split,imgsz=imgsz,batch=1,device='cpu',candidate_confidence=.001,
                            nms_iou=.7,map_iou_thresholds=[round(float(x),2) for x in np.arange(.5,1,.05)],
                            precision_recall_operating_point='Ultralytics validation maximum smoothed F1 point; not confidence .001',
                            confusion_matrix=dict(confidence=.25,matching_iou=.45,note='Ultralytics plot defaults; separate from AP and NMS'),
                            matching='Official one-to-one IoU matching at 0.50 through 0.95')
    from ultralytics.utils.metrics import smooth
    metrics['settings']['confidence_at_maximum_f1']=float(result.box.px[smooth(result.box.f1_curve.mean(0),.1).argmax()])
    if split=='test' and Path(data).resolve() in ((ROOT/'.cache/data_runtime.yaml').resolve(),(DATA/'data.yaml').resolve()):
        metrics['fixed_confidence_iou50']=fixed_iou_metrics(model,imgsz,threads)
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics

def compare(imgsz=320,threads=2):
    import pandas as pd
    validate_dataset()
    data=runtime_yaml()
    results={}
    for label,model in [('pretrained',ROOT/'yolo11n.pt'),('fine_tuned',ROOT/'models/best.pt')]:
        results[label]=evaluate(model,data,'test',ROOT/'reports/evaluation'/label,imgsz,threads)
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'evaluation_comparison.json').write_text(json.dumps(results,indent=2))
    rows=[dict(metric=key,pretrained=results['pretrained'][key],fine_tuned=results['fine_tuned'][key])
          for key in ('precision','recall','f1','map50','map50_95')]
    pd.DataFrame(rows).to_csv(out/'evaluation_comparison.csv',index=False)
    table='| Metric | Pretrained | Fine-tuned |\n|---|---:|---:|\n'
    table+='\n'.join(f"| {r['metric']} | {r['pretrained']:.4f} | {r['fine_tuned']:.4f} |" for r in rows)
    delta=results['fine_tuned']['map50_95']-results['pretrained']['map50_95']
    text=f'# Held-out evaluation\n\n{table}\n\nFine-tuning mAP@50:95 change: {delta:+.4f}.\n'
    test_count=json.loads((DATA/'quality_report.json').read_text())['splits']['test']['images']
    text+=f'This is a {test_count}-image grouped test subset, not a production accuracy estimate. No test data was used for checkpoint selection. Precision/recall/F1 are at the same Ultralytics validation operating point. NMS IoU is .70; AP matching spans .50:.95, candidate confidence .001. Single class: person.\n'
    text+='\nA small CPU experiment can improve or degrade a pretrained model. Small data, cropped-person annotation policy, head re-initialization and limited optimization are plausible causes; they are not proven by this run. Do not tune on this test result. Collect independent site data before deployment.\n'
    (ROOT/'EVALUATION_REPORT.md').write_text(text)
    print(json.dumps(results,indent=2),flush=True)
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compare',action='store_true')
    p.add_argument('--data');p.add_argument('--model',default='models/best.pt')
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--output',default='reports/evaluation/custom')
    p.add_argument('--imgsz',type=int,default=320);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['cpu'],default='cpu')
    a=p.parse_args()
    if a.compare:compare(a.imgsz,a.threads)
    else:print(json.dumps(evaluate(a.model,Path(a.data) if a.data else runtime_yaml(),a.split,a.output,a.imgsz,a.threads),indent=2))
