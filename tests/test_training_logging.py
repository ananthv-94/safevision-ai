import json
from pathlib import Path
import pytest
def test_training_logger_records_root_artifacts_and_epoch_metrics(tmp_path,monkeypatch):
    pytest.importorskip('mlflow')
    import safevision.training_log as training_log
    from mlflow.tracking import MlflowClient
    output=tmp_path/'completed'
    output.mkdir()
    (output/'results.csv').write_text('epoch,train/box_loss,metrics/mAP50(B)\n1,1.2,0.5\n')
    (output/'training_config.json').write_text('{"device":"cpu"}')
    (output/'training_summary.json').write_text('{"epochs_completed":1,"mlflow":{"old":"receipt"}}')
    monkeypatch.setattr(training_log,'DEFAULT_STORE',tmp_path/'tracking')
    result=training_log.log_training(output,{'device':'cpu','dataset_version':'unit-test-fixture'})
    assert result['status']=='logged',result
    client=MlflowClient(tracking_uri=(tmp_path/'tracking').resolve().as_uri())
    run=client.get_run(result['run_id'])
    assert run.info.status=='FINISHED' and run.data.params['device']=='cpu'
    assert run.data.metrics['train/box_loss']==1.2
    assert run.data.metrics['metrics/mAP50_B_']==.5
    artifacts={Path(a.path).name for a in client.list_artifacts(run.info.run_id,'training')}
    assert {'results.csv','training_config.json','training_summary.json'}<=artifacts
    summary=json.loads(Path(client.download_artifacts(run.info.run_id,'training/training_summary.json')).read_text())
    assert 'mlflow' not in summary
