from evaluate import iou

def test_iou():
    assert iou([0,0,2,2],[0,0,2,2])==1
    assert iou([0,0,1,1],[2,2,3,3])==0
    assert abs(iou([0,0,2,2],[1,1,3,3])-1/7)<1e-8
