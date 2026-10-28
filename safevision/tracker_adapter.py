"""Numpy detections for Ultralytics trackers; ONNX never executes a torch model."""
from types import SimpleNamespace
import numpy as np

class NumpyDetections:
    def __init__(self,data):self.data=np.asarray(data,dtype=np.float32).reshape(-1,6)
    def __len__(self):return len(self.data)
    def __getitem__(self,key):return NumpyDetections(self.data[key])
    @property
    def conf(self):return self.data[:,4]
    @property
    def cls(self):return self.data[:,5]
    @property
    def xyxy(self):return self.data[:,:4]
    @property
    def xywh(self):
        boxes=self.xyxy
        return np.c_[(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]]

def create_tracker(name,fps):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.bot_sort import BOTSORT
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                         track_buffer=30,match_thresh=.8,fuse_score=True,
                         gmc_method='sparseOptFlow',proximity_thresh=.5,appearance_thresh=.8,
                         with_reid=False,model='auto')
    return (BYTETracker if name=='bytetrack' else BOTSORT)(args,frame_rate=round(fps))
