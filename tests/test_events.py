import csv
import pytest
from safevision.events import polygon, inside, Incidents

def test_footpoint_and_boundary():
    z=polygon([[.2,.2],[.8,.2],[.8,.8],[.2,.8]],101,101)
    assert inside([0,0,40,20],z)
    assert inside([40,0,60,50],z)
    assert not inside([40,0,60,90],z)

@pytest.mark.parametrize('points', [[], [[0,0],[1,1]], [[0,0],[2,0],[1,1]], [[0,0],[1,1],[0,1],[1,0]]])
def test_invalid_zone(points):
    with pytest.raises(ValueError):
        polygon(points,100,100)

def test_continuous_occupancy_and_reentry():
    log=Incidents(cooldown=5,exit_grace=.5)
    assert log.update(1,True,0)
    assert not log.update(1,True,10) # no periodic alarm while continuously inside
    log.update(1,False,11)
    log.update(1,False,11.6)
    assert log.update(1,True,12)
    assert len(log.rows)==2

def test_jitter_and_cooldown():
    log=Incidents(cooldown=5)
    log.update(1,True,0)
    log.update(1,False,1)
    log.update(1,True,1.1)
    assert len(log.rows)==1
    log.update(1,False,2)
    log.update(1,False,2.6)
    log.update(1,True,3)
    assert len(log.rows)==1
    log.update(1,True,20)
    assert len(log.rows)==1

def test_independent_ids_and_csv(tmp_path):
    log=Incidents()
    log.update(1,True,1.2345)
    log.update(2,True,1.2345)
    path=tmp_path/'events.csv'
    log.save(path)
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==2 and rows[0]['video_seconds']=='1.234'
    assert rows[1]['track_id']=='2'

def test_empty_report_has_header(tmp_path):
    p=tmp_path/'empty.csv'
    Incidents().save(p)
    assert list(csv.DictReader(p.open()))==[]
