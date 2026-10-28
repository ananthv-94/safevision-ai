"""Optional local MLflow tracking, independent of global active-run state."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / 'mlruns'
EXPERIMENT = 'SafeVision video analytics'
METRICS = ('frames', 'source_fps', 'unique_track_ids', 'incidents', 'processing_fps',
           'inference_ms', 'processing_seconds', 'encoding_seconds')

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def log_result(output, source=None, store=None):
    """Log a completed result. A tracking failure never invalidates the video."""
    output = Path(output).resolve()
    client, run_id = None, None
    started = time.perf_counter()
    try:
        from mlflow.tracking import MlflowClient
        directory = Path(store or DEFAULT_STORE).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        uri = directory.as_uri()
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(EXPERIMENT)
            except Exception:
                experiment = client.get_experiment_by_name(EXPERIMENT)
                if experiment is None:
                    raise
                experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        run = client.create_run(experiment_id, tags={
            'mlflow.runName': f"{summary['tracker']}-{output.name}",
            'safevision.stage': 'inference',
            'safevision.accuracy_evaluated': 'false',
        })
        run_id = run.info.run_id
        params = {key: summary[key] for key in
                  ('tracker', 'confidence', 'model', 'device', 'cooldown_seconds')}
        params.update(zone=json.dumps(summary['zone']), tracking_candidate_confidence=0.1,
                      image_size=640, person_class_id=0, footpoint='bottom_center',
                      source_name=Path(str(source)).name if source is not None else 'not_provided')
        for key in ('inference_engine','inference_imgsz','cpu_threads'):
            if key in summary: params[key]=summary[key]
        if source is not None and not isinstance(source, int) and Path(source).is_file():
            params['source_sha256'] = _sha256(source)
        model = Path(summary['model'])
        if model.is_file():
            params['model_sha256'] = _sha256(model)
        for key, value in params.items():
            client.log_param(run_id, key, value)
        for key in METRICS:
            client.log_metric(run_id, key, float(summary[key]))
        for key, value in summary.get('environment', {}).items():
            client.set_tag(run_id, f'environment.{key}', str(value))
        for name in ('summary.json', 'incidents.csv', 'preview.jpg', 'annotated.mp4'):
            path = output / name
            if path.is_file():
                client.log_artifact(run_id, str(path), artifact_path='outputs')
        client.set_terminated(run_id, status='FINISHED')
        result = dict(status='logged', run_id=run_id, experiment_id=experiment_id,
                      tracking_uri=uri, logging_seconds=time.perf_counter() - started)
    except Exception as exc:
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status='FAILED')
            except Exception:
                pass
        result = dict(status='failed', run_id=run_id, error=f'{type(exc).__name__}: {exc}',
                      logging_seconds=time.perf_counter() - started)
    try:
        (output / 'mlflow.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    except OSError:
        pass
    return result
