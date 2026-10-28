> Historical verification before the CPU training upgrade. Current status: UPGRADE_VERIFICATION.md.

# Verified on 24 October 2026

## Completed

- Windows 11, Python 3.12.14, AMD CPU, PyTorch 2.7.1+cpu,
  Ultralytics 8.3.228, OpenCV 4.11.0, Streamlit 1.50.0.
- Real warehouse stock footage by Tiger Lily / Pexels; attribution and license link in
  `demo/ATTRIBUTION.md`. The polygon is staged for this software demo.
- YOLO11 nano weights included. ByteTrack and stock BoT-SORT each ran on the same first
  250 frames of the 1280×720, 25 FPS source (10 seconds), confidence 0.35, default polygon.
- ByteTrack: **11.11 processing FPS**, **51.33 ms mean inference**, 22.50 s processing,
  2.31 s H.264 encoding, **4 unique track IDs**, **3 zone-entry incidents**.
- BoT-SORT: **9.66 processing FPS**, **51.09 ms mean inference**, 25.88 s processing,
  1.35 s H.264 encoding, **4 unique track IDs**, **3 zone-entry incidents**.
- A separate actual dashboard Start-button run completed all 250 frames: 12.36 processing
  FPS, 53.27 ms mean inference, 4 IDs and 3 events. Run-to-run timing varies.
- `demo/annotated.mp4` was decoded end to end: 250 frames, 1280×720. Incident CSV row count
  and timestamp range match the summary. SHA-256:
  `8793c536514c9f7a922698fe00bbef74dcf55ebd4fc4b0d6eb37ea98e09d16bc`.
- Browser video decoder reported 1280-pixel width, 10-second duration, ready state 4.
  Playback advanced to 8.54 seconds with `paused=false`.
- **12 tests passed** on the final code. Coverage: footpoint geometry/boundaries, invalid
  polygons, continuous occupancy, exit jitter, cooldown/re-entry, independent IDs, CSV
  headers/contents, bounding-box IoU, dashboard startup and invalid-zone error display,
  and invalid-video failure without a success summary.
- 66 locally installed dependency distributions audited: no unsatisfied runtime dependency
  requirements. Assembled PyTorch wheel checked against PyPI's SHA-256 digest.
- Imports verified in the prepared local `.venv`; its Streamlit server started successfully.
  A separate CLI/dashboard testing environment was used for the automated tests.

The tracker throughput figures above are single-run local measurements, not a broad
benchmark or a guarantee of 25 FPS real-time operation. No identity labels were available:
four IDs do not establish four distinct workers, and ID-switch counts are not claimed.

## Environment notes

The bundled Python initially lacked vision packages, and its virtual environment could
not bootstrap pip. Dependencies were installed in the workspace using the bundled pip.
Windows temporary-folder permissions required a test/install harness using workspace
temporary files; that host workaround is outside the distributed application. A prepared
local `.venv` references `work/packages`; retain `work/` on this computer. A fresh ZIP uses
the normal Windows setup in the README and does not contain a machine-specific environment.
Large downloads required retries. The lower-resolution source URL was denied; the official
full-resolution Pexels download succeeded and was resized/encoded to a 720p H.264 sample.

## Implemented but not empirically validated here

- Webcam capture and CUDA device selection are available but were not exercised.
- `evaluate.py` provides IoU and Ultralytics validation for precision, recall, F1,
  mAP@50 and mAP@50:95. No annotated evaluation dataset was supplied, so no detection
  accuracy measurements are included.
- ONNX export helper is supplied; export/runtime comparison was not run.
- TensorRT, PPE detection, RF-DETR and segmentation are deferred. MLflow was added and verified on 25 October (below).

## Interview demonstration

1. On this computer, double-click `START_DEMO.bat`; open the displayed local address.
2. Play the preloaded 10-second annotated warehouse demonstration.
3. Point out footpoints, tracker IDs, staged zone, red alerts and the incident table.
4. Export the CSV. Explain its video-relative timestamps and ID-fragmentation limitation.
5. If time permits, process the included video again with the default 250-frame limit.
6. Use `INTERVIEW_GUIDE.md` for the architecture and two-minute presentation script.

The ZIP includes source, tests, weights, original resized source, both trackers' annotated
videos, measured summaries, incident CSVs, screenshots, and interview documentation.


## MLflow addition — 25 October 2026

The version before MLflow was copied and verified byte-for-byte as
`SafeVision_AI_before_MLflow_2026-10-25.zip` (SHA-256
`7cbf5d98c2b89b54422edeb1ae9caffca3485222240e971be6872013502bf3d2`).
An extracted copy with a separate local launch environment is retained alongside it.
The original vision dependency directory was kept separate from the new MLflow dependencies.

MLflow 3.5.1 is installed and integrated into the CLI and dashboard. Three real
completed video outputs were logged: the original ByteTrack and BoT-SORT results,
plus a fresh 250-frame YOLO + ByteTrack run with automatic MLflow logging enabled.
The fresh run measured 11.82 processing FPS and 54.11 ms inference, produced 4 IDs
and 3 events, and logged successfully. MLflow overhead was measured separately
(4.62 seconds in this run) and excluded from processing FPS.

The real MLflow client retrieved all three FINISHED runs, checked parameters and
metrics against their source summaries, and verified saved video artifact hashes.
See `demo/mlflow_validation.json`. All 111 installed core/optional distributions
passed dependency checks. `requirements-mlflow-tested.txt` records the optional set.

The final automated suite has 15 passing tests. Added tests verify real MLflow
parameters/metrics/artifact logging, experiment reuse with distinct run IDs,
graceful handling of missing MLflow, and tracking-storage failure without losing
the completed video results.

MLflow's actual application routes were tested in-process: health, version, UI
HTML, experiment search and run search returned HTTP 200; run search returned all
three real runs. See `demo/mlflow_http_validation.json`.
The UI server initializes with one worker; its default multi-worker mode failed
on this Windows execution host. `START_MLFLOW.bat` includes `--workers 1`.

Browser access to background services launched through this execution environment
timed out, including the Streamlit service. No new MLflow browser screenshot is
claimed. Launch `START_DEMO.bat` and `START_MLFLOW.bat` directly on this computer
to use the local interfaces. The existing dashboard screenshots date from the
original verified demonstration on 24 October and predate the MLflow controls.

Local file-store artifacts contain absolute locations. The updated ZIP includes
output files, run receipts and validation reports, but excludes the machine-local
`mlruns/` directory. On extraction elsewhere, re-log the supplied output folders
using `track_run.py` to recreate experiment history.
