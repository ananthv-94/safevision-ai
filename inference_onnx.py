"""Direct ONNX Runtime CPU inference. This module never imports PyTorch."""
import argparse,json,sys
from pathlib import Path
import cv2
from safevision.engines import OnnxEngine
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--model',default='models/best.onnx')
    p.add_argument('--output',default='reports/onnx_prediction.jpg');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();image=cv2.imread(a.image)
    if image is None:p.error('Cannot decode image')
    engine=OnnxEngine(a.model,a.threads);detections,ms=engine.detect(image,confidence=.25)
    for x1,y1,x2,y2,score,cls in detections:
        cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(60,230,90),2)
        cv2.putText(image,f'person {score:.2f}',(int(x1),max(15,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(60,230,90),1)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(a.output,image)
    assert 'torch' not in sys.modules,'Unexpected PyTorch import in standalone ONNX path'
    result=dict(provider=engine.session.get_providers(),inference_ms=ms,detections=detections.tolist(),torch_imported=False)
    Path(a.output).with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
