import json
from pathlib import Path
import numpy as np
import pytest
from safevision.engines import preprocess,decode,box_iou
from safevision.tracker_adapter import NumpyDetections
from safevision.workflow import DATA,validate_dataset
def test_letterbox_contract():
    image=np.zeros((50,100,3),dtype=np.uint8)
    tensor,meta=preprocess(image,100)
    assert tensor.shape==(1,3,100,100) and tensor.dtype==np.float32
    assert meta==(1,0,25,100,50)
    assert np.allclose(tensor[0,:,0,0],114/255)
    assert np.all(tensor[0,:,30,10]==0)
def test_nms_and_reverse_letterbox():
    raw=np.array([[[50,51,85],[50,50,50],[20,20,10],[20,20,10],[.9,.8,.7]]],dtype=np.float32)
    result=decode(raw,(1,0,25,100,50),.25,.5)
    assert len(result)==2
    assert np.allclose(result[0,:4],[40,15,60,35])
    assert np.isclose(result[0,4],.9)
def test_no_predictions():
    assert decode(np.zeros((1,5,3)),(1,0,0,100,100)).shape==(0,6)
def test_iou_and_numpy_tracker_contract():
    assert box_iou([0,0,10,10],np.array([[0,0,10,10],[10,10,20,20]])).tolist()==[1,0]
    boxes=NumpyDetections([[0,0,10,20,.8,0],[20,20,30,40,.6,0]])
    assert np.allclose(boxes.xywh,[[5,10,10,20],[25,30,10,20]])
    assert len(boxes[np.array([True,False])])==1
@pytest.mark.skipif(not (DATA/'manifest.json').exists(),reason='Prepare the dataset first')
def test_prepared_dataset_has_disjoint_groups_and_valid_boxes():
    report=validate_dataset()
    assert report['images']==sum(report['splits'].values())
    assert report['boxes']>0 and all(report['splits'].values())
def test_tracker_persists_id_with_numpy_detections():
    from safevision.tracker_adapter import create_tracker
    tracker=create_tracker('bytetrack',25)
    detections=NumpyDetections([[10,10,50,100,.9,0]])
    first=tracker.update(detections,np.zeros((120,120,3),np.uint8))
    second=tracker.update(detections,np.zeros((120,120,3),np.uint8))
    assert len(first)==len(second)==1 and first[0,4]==second[0,4]
