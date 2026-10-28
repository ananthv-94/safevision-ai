"""Training records use the same local store as existing inference tracking."""
import json,re
from pathlib import Path
from .experiment import DEFAULT_STORE

def log_training(output,config,record_type='completed_training_output'):
    client=None;run_id=None
    try:
        import pandas as pd
        from mlflow.tracking import MlflowClient
        client=MlflowClient(tracking_uri=DEFAULT_STORE.resolve().as_uri())
        name='SafeVision training'
        experiment=client.get_experiment_by_name(name)
        eid=experiment.experiment_id if experiment else client.create_experiment(name)
        run=client.create_run(eid,tags={'mlflow.runName':'yolo11n-cpu-person','safevision.stage':'training','safevision.record_type':record_type})
        run_id=run.info.run_id
        for key,value in config.items():
            client.log_param(run_id,key,json.dumps(value) if isinstance(value,dict) else value)
        history=pd.read_csv(Path(output)/'results.csv');history.columns=history.columns.str.strip()
        for _,row in history.iterrows():
            for key,value in row.items():
                if key not in ('epoch','time') and pd.notna(value):
                    client.log_metric(run_id,re.sub(r'[^a-zA-Z0-9_/ .-]','_',key),float(value),step=int(row['epoch']))
        for path in Path(output).rglob('*'):
            if path.is_file() and path.suffix in ('.png','.jpg','.csv','.json','.yaml','.pt'):
                # Retain best checkpoint; last.pt is locally available, not duplicated in tracking.
                if path.name in ('last.pt','mlflow_training.json','training_summary.json'):continue
                client.log_artifact(run_id,str(path),artifact_path='training' if str(path.relative_to(output).parent)=='.' else 'training/'+str(path.relative_to(output).parent).replace('\\','/'))
        summary_path=Path(output)/'training_summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            summary.pop('mlflow',None)
            client.log_dict(run_id,summary,'training/training_summary.json')
        client.set_terminated(run_id,status='FINISHED')
        receipt=dict(status='logged',run_id=run_id,experiment_id=eid,tracking_uri=DEFAULT_STORE.resolve().as_uri())
    except Exception as exc:
        if client and run_id:
            try:client.set_terminated(run_id,status='FAILED')
            except Exception:pass
        receipt=dict(status='failed',error=f'{type(exc).__name__}: {exc}',run_id=run_id)
    (Path(output)/'mlflow_training.json').write_text(json.dumps(receipt,indent=2))
    return receipt
