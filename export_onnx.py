"""Export the genuinely trained CPU model, then verify graph and runtime."""
import argparse,json,hashlib
from pathlib import Path
import safevision
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='models/best.pt')
    p.add_argument('--imgsz',type=int,default=320);a=p.parse_args()
    from ultralytics import YOLO
    import onnx,numpy as np
    from safevision.engines import OnnxEngine
    model=Path(a.model)
    exported=Path(YOLO(str(model)).export(format='onnx',imgsz=a.imgsz,batch=1,device='cpu',
                                        dynamic=False,simplify=False,opset=17,nms=False,half=False))
    graph=onnx.load(str(exported));onnx.checker.check_model(graph)
    engine=OnnxEngine(exported)
    output=engine.forward(np.zeros((1,3,a.imgsz,a.imgsz),np.float32))
    report=dict(status='verified',model=str(exported),provider=engine.session.get_providers(),
                input_shape=[1,3,a.imgsz,a.imgsz],output_shape=list(output.shape),
                source_checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                precision='FP32',opset=17,graph_check='passed',model_bytes=exported.stat().st_size)
    Path('reports').mkdir(exist_ok=True);Path('reports/onnx_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
