# SafeVision AI — CPU learning and deployment workflow

A working YOLO11 **Nano** industrial person-monitoring project, extended with a small
supervised training experiment, held-out evaluation, MLflow training records and
direct ONNX Runtime CPU inference. The original detector remains the default:
the small fine-tuning experiment reduced test accuracy. Read the measured reports
before selecting the experimental fine-tuned model.

## Open the prepared project on this computer

Double-click `START_DEMO.bat`. The dashboard shows the existing warehouse demo,
incident CSV downloads, model/engine selection, and a **Model Evaluation &
Benchmarking** section populated from actual report files. Double-click
`START_MLFLOW.bat` to view experiment history at http://127.0.0.1:5000.

The local `.venv` uses packages in this workspace's `work/` directory; keep that
directory in place. ZIPs exclude the machine-specific environment and live MLflow
store. Use the installation below on a new computer. Browser connections to
tool-launched background services timed out in this execution environment; the
dashboard is verified with Streamlit AppTest and the launchers are supplied for
normal Windows use. Existing dashboard screenshots predate these new controls.

## Windows installation — Python 3.11 or 3.12, CPU

Open this folder in VS Code and run in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt -r requirements-mlflow-tested.txt -r requirements-cpu.txt
```

The tested core and optional dependency versions are pinned separately. CPU is
explicit in training/evaluation and in ONNX's execution provider. No paid API,
credentials or GPU is needed. The Nano model is used throughout.

## Reproduce the workflow

The dataset, trained checkpoint, ONNX file, plots and measured reports are already
included. These commands reproduce or replace generated results.

### 1. Dataset preparation and validation

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --count 120 --seed 42
.\.venv\Scripts\python.exe -c "from safevision.workflow import validate_dataset; print(validate_dataset())"
```

This downloads a small subset from the authors' public SH17 sources, keeps existing
human person annotations, and produces `datasets/safevision/`. The source is
CC BY-NC-SA 4.0; photos retain Pexels terms. See its license/source file, manifest
and quality report. This is a noncommercial educational experiment.

Groups are inferred conservatively from nearby source photo IDs and perceptual
duplicate checks, not invented camera metadata. Exact/group overlap checks pass;
undetected scene overlap remains possible. This is not a validated site-level
production split. Original human class-0 labels are retained; no pseudo-labels.

### 2. Fine-tune YOLO11 Nano on CPU

```powershell
.\.venv\Scripts\python.exe train.py --epochs 10 --imgsz 320 --batch 4 --threads 2 --output training/my_cpu_run
```

Ten CPU epochs were actually completed for the included experiment.
A new output name avoids overwriting the included training run. The new selected
checkpoint is also copied to `models/best.pt`; rerun evaluation/export after
training a different checkpoint. Model selection uses validation only.

### 3. View MLflow training experiments

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000 --workers 1
```

After extracting on a new computer, recreate history from the included results
without retraining or remeasuring:

    ./.venv/Scripts/python.exe log_training.py training/cpu_person
    ./.venv/Scripts/python.exe track_run.py demo --source demo/warehouse.mp4
    ./.venv/Scripts/python.exe track_run.py demo/botsort --source demo/warehouse.mp4

Select **SafeVision training**. Training logs hyperparameters, dataset version and
sizes, per-epoch training/validation losses and metrics, plots and the best model.
Existing **SafeVision video analytics** inference logging still works. MLflow
failures are recorded without discarding completed video/training results.

### 4. Evaluate both models on the held-out test set

```powershell
.\.venv\Scripts\python.exe evaluate.py --compare --imgsz 320 --threads 2
```

See `EVALUATION_REPORT.md`, `reports/evaluation_comparison.csv` and
`reports/evaluation/`. Both models use the same test set and official Ultralytics
settings. Reports include per-class AP and a separate fixed-confidence IoU analysis.
Do not tune hyperparameters on this test comparison.

### 5. Export and verify ONNX

```powershell
.\.venv\Scripts\python.exe export_onnx.py --model models/best.pt --imgsz 320
```

Writes `models/best.onnx`, validates the ONNX graph and runs a CPU inference.
The export is fixed batch 1, 320×320, FP32, opset 17, with NMS outside the graph.

### 6. Direct ONNX Runtime inference

```powershell
.\.venv\Scripts\python.exe inference_onnx.py datasets/safevision/images/test/pexels-photo-3831159.jpg --model models/best.onnx
```

This uses ONNX Runtime directly, with numpy/OpenCV preprocessing and NMS. It never
imports PyTorch or Ultralytics. A guarded run with both imports deliberately
blocked succeeded; see `reports/torch_free_proof.json`.

For inference-only installation, use `requirements-onnx.txt` in a separate
environment; training/export tools are not needed to execute the supplied ONNX.

### 7. Benchmark the same fine-tuned model on CPU

```powershell
.\.venv\Scripts\python.exe benchmark.py --threads 2 --imgsz 320 --warmup 10 --repeats 100 --trials 3 --video demo/warehouse.mp4 --video-frames 100
```

See `BENCHMARK_REPORT.md`, `benchmark_results.csv` and `reports/benchmark/`.
Each runtime runs in separate processes, sequentially, with alternating trial
order. Both use FP32, batch 1, the same images, square letterboxing and numpy NMS.
Loading and warmup are excluded from steady-state latency. Peak process RAM
includes libraries and inputs, and is not the model's memory footprint.
The raw per-inference timings and prediction agreement are retained.

### 8. Monitor video with the trained detector

```powershell
.\.venv\Scripts\python.exe run.py demo/warehouse.mp4 --model models/best.pt --engine pytorch --output runs/fine_pytorch --max-frames 250 --mlflow
.\.venv\Scripts\python.exe run.py demo/warehouse.mp4 --model models/best.onnx --engine onnx --tracker botsort --output runs/fine_onnx --max-frames 250 --mlflow
```

Both engines support the existing ByteTrack/BoT-SORT, polygon footpoints, event
cooldown, CSV export and H.264 output. In the integrated tracking path,
Ultralytics tracker libraries import torch; ONNX model execution itself remains
direct CPUExecutionProvider. Use the standalone script for a Torch-free process.

### 9. Launch the dashboard and run tests

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Monitoring behavior

Zones are convex polygons of normalized coordinates. The bottom-center footpoint
is tested with OpenCV; the boundary counts as inside. First observation inside
counts as entry. Continuous occupancy generates one event. Exit requires observed
outside footpoints over 0.5 seconds; re-entry uses a per-ID cooldown. Missing or
low-confidence detections are unknown, not exits. IDs are trajectories, not
verified unique people. ID fragmentation can duplicate incidents.

The tracking candidate threshold is .10. The dashboard's display/alert threshold
is separate. Source-video seconds are used in the CSV. Processing FPS includes
decode, detection, tracking, drawing and intermediate writing, but excludes warmup,
model loading, final H.264 encoding and optional MLflow logging.

Uploaded videos and results stay under `runs/`. Webcam support uses the computer's
local camera and requires a finite frame limit; it remains untested on this host.

## Files and evidence

- `datasets/safevision/`: images, YOLO labels, manifest, quality report, annotation previews.
- `training/cpu_person/`: actual training CSV, configuration, curves, best/last checkpoints.
- `models/`: genuine fine-tuned best.pt and verified best.onnx.
- `reports/`: measured evaluation, benchmark, ONNX and regression evidence.
- `safevision/`: original events/pipeline plus engine and tracker adapters.
- `TRAINING_GUIDE.md`, `EVALUATION_REPORT.md`, `BENCHMARK_REPORT.md`,
  `INTERVIEW_GUIDE.md`, `UPGRADE_VERIFICATION.md`: methods, results and limits.

MLflow's local file store uses absolute artifact paths. ZIPs contain portable
results and receipts but exclude the live store. Recreate inference history after
relocation with `python track_run.py demo --source demo/warehouse.mp4`. A new
training run recreates its training history. The included source/checkpoints
remain usable without the old tracking database.

Earlier ZIPs, including the version before this training upgrade, are retained
alongside the latest ZIP. See the parent `VERSIONS.md`.

Ultralytics code/weights retain their own license requirements:
https://www.ultralytics.com/license . Dataset and sample-video notices are included.

After moving the ZIP, recreate training history without retraining using: python log_training.py training/cpu_person . This logs completed outputs and is tagged as replayed history.
