"""Shared FP32 image contract. ONNX inference imports no PyTorch."""
import time
import cv2
import numpy as np

def preprocess(image,size=320):
    h,w=image.shape[:2];ratio=min(size/h,size/w)
    nw,nh=round(w*ratio),round(h*ratio)
    left=round((size-nw)/2-.1);top=round((size-nh)/2-.1)
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,top,size-nh-top,left,size-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
    return tensor,(ratio,left,top,w,h)

def box_iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    lo=np.maximum(a[:2],b[:,:2]);hi=np.minimum(a[2:4],b[:,2:4])
    inter=np.maximum(hi-lo,0).prod(axis=1)
    union=np.maximum(a[2]-a[0],0)*np.maximum(a[3]-a[1],0)+np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)-inter
    return inter/np.maximum(union,1e-9)

def decode(raw,meta,confidence=.25,iou_threshold=.7,max_det=300):
    # YOLO11 export: [batch, 4 + class_count, anchor_count], no objectness channel.
    raw=np.asarray(raw)
    if raw.ndim!=3 or raw.shape[0]!=1 or raw.shape[1]<5:raise ValueError('Expected YOLO11 raw detection output')
    rows=raw[0].T
    classes=rows[:,4:].argmax(1);scores=rows[:,4:].max(1)
    keep=(classes==0)&(scores>=confidence)
    rows=rows[keep];scores=scores[keep];classes=classes[keep]
    if not len(rows):return np.empty((0,6),dtype=np.float32)
    boxes=np.c_[rows[:,:2]-rows[:,2:4]/2,rows[:,:2]+rows[:,2:4]/2]
    order=np.argsort(-scores,kind='stable')[:30000];selected=[]
    while len(order) and len(selected)<max_det:
        index=order[0];selected.append(index)
        order=order[1:][box_iou(boxes[index],boxes[order[1:]])<=iou_threshold]
    boxes=boxes[selected]
    ratio,left,top,w,h=meta
    boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/ratio
    boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/ratio
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
    return np.c_[boxes,scores[selected],classes[selected]].astype(np.float32)

class OnnxEngine:
    def __init__(self,model,threads=2):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        inp=self.session.get_inputs()[0]
        if len(inp.shape)!=4 or not isinstance(inp.shape[2],int) or inp.shape[2]!=inp.shape[3]:
            raise ValueError('Export a fixed square batch-one YOLO11 FP32 model')
        self.size=inp.shape[2];self.input_name=inp.name
        if inp.shape[0]!=1 or inp.type!='tensor(float)':raise ValueError('Expected batch 1 FP32 input')
    def forward(self,tensor):
        return self.session.run(None,{self.input_name:tensor})[0]
    def detect(self,image,confidence=.1,iou_threshold=.7):
        tensor,meta=preprocess(image,self.size);start=time.perf_counter()
        raw=self.forward(tensor);ms=(time.perf_counter()-start)*1000
        return decode(raw,meta,confidence,iou_threshold),ms

class TorchEngine:
    def __init__(self,model,threads=2,size=320):
        import torch
        from ultralytics import YOLO
        self.torch=torch;torch.set_num_threads(threads)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        self.model=YOLO(str(model)).model.float().cpu().eval()
        self.model.fuse(verbose=False)
        torch.set_num_threads(threads)
        self.size=size
    def forward(self,tensor):
        with self.torch.inference_mode():
            raw=self.model(self.torch.from_numpy(tensor))
        if isinstance(raw,(tuple,list)):raw=raw[0]
        return raw.detach().numpy()
    def detect(self,image,confidence=.1,iou_threshold=.7):
        tensor,meta=preprocess(image,self.size);start=time.perf_counter()
        raw=self.forward(tensor);ms=(time.perf_counter()-start)*1000
        return decode(raw,meta,confidence,iou_threshold),ms
