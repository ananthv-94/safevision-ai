import argparse
import json
from safevision.pipeline import process

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='YOLO person tracking and zone monitoring')
    p.add_argument('source', help='Local video path, or camera:0')
    p.add_argument('--output', default='runs/latest')
    p.add_argument('--tracker', choices=['bytetrack','botsort'],default='bytetrack')
    p.add_argument('--confidence',type=float,default=.35)
    p.add_argument('--model',default='yolo11n.pt')
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--zone',help='JSON normalized convex polygon')
    p.add_argument('--mlflow',action='store_true',help='Log results to the local MLflow store')
    p.add_argument('--engine',choices=['ultralytics','pytorch','onnx'],default='ultralytics')
    p.add_argument('--imgsz',type=int,default=320)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    source=int(a.source.split(':')[1]) if a.source.startswith('camera:') else a.source
    if isinstance(source,int) and a.max_frames<=0:
        p.error('Webcam input requires a positive --max-frames limit.')
    print(json.dumps(process(source,a.output,tracker=a.tracker,confidence=a.confidence,
          model_path=a.model,device=a.device,max_frames=a.max_frames,
          zone=json.loads(a.zone) if a.zone else None,track_experiment=a.mlflow,engine=a.engine,imgsz=a.imgsz,threads=a.threads),indent=2))
