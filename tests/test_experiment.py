import builtins
import json
from pathlib import Path
import pytest
from safevision.experiment import log_result, METRICS

def fixture_output(tmp_path):
    output=tmp_path/'result'
    output.mkdir()
    summary=dict(frames=10, source_fps=25, unique_track_ids=2, incidents=1,
                 processing_fps=12.5, inference_ms=50, processing_seconds=.8,
                 encoding_seconds=.1, tracker='bytetrack', confidence=.35,
                 model='test-model.pt', device='cpu', zone=[[0,0],[1,0],[1,1]],
                 cooldown_seconds=5, environment={'python':'test-fixture'})
    (output/'summary.json').write_text(json.dumps(summary))
    (output/'incidents.csv').write_text('video_seconds,track_id,violation_type,zone_id\n0,1,restricted_zone_entry,zone_1\n')
    return output,summary

def test_missing_mlflow_preserves_outputs(tmp_path,monkeypatch):
    output,summary=fixture_output(tmp_path)
    before=(output/'summary.json').read_bytes()
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name.startswith('mlflow'):
            raise ImportError('MLflow deliberately unavailable in this test')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    result=log_result(output,store=tmp_path/'store')
    assert result['status']=='failed'
    assert 'ImportError' in result['error']
    assert (output/'summary.json').read_bytes()==before
    assert (output/'incidents.csv').exists()

def test_real_mlflow_records_metrics_params_and_artifacts(tmp_path):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    output,summary=fixture_output(tmp_path)
    store=tmp_path/'store'
    first=log_result(output,store=store)
    assert first['status']=='logged',first
    client=MlflowClient(tracking_uri=store.resolve().as_uri())
    run=client.get_run(first['run_id'])
    assert run.info.status=='FINISHED'
    assert run.data.params['tracker']=='bytetrack'
    assert run.data.params['footpoint']=='bottom_center'
    for key in METRICS:
        assert run.data.metrics[key]==summary[key]
    artifacts={Path(a.path).name for a in client.list_artifacts(first['run_id'],'outputs')}
    assert {'summary.json','incidents.csv'}<=artifacts
    second=log_result(output,store=store)
    assert second['status']=='logged'
    assert second['experiment_id']==first['experiment_id']
    assert second['run_id']!=first['run_id']

def test_failed_artifact_logging_is_reported_without_losing_video_results(tmp_path,monkeypatch):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    output,_=fixture_output(tmp_path)
    def fail(*args,**kwargs):
        raise OSError('Test storage failure')
    monkeypatch.setattr(MlflowClient,'log_artifact',fail)
    result=log_result(output,store=tmp_path/'store')
    assert result['status']=='failed'
    assert (output/'summary.json').exists()
    client=MlflowClient(tracking_uri=(tmp_path/'store').resolve().as_uri())
    assert client.get_run(result['run_id']).info.status=='FAILED'

