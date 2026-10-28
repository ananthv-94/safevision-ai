# CPU upgrade verification — 25 October 2026

## Final checklist

| Requested task | Status | Executed evidence |
|---|---|---|
| Annotated dataset | Completed as a small pilot | 120 images, 234 boxes, 84/18/18 grouped split; all file/label/hash checks pass |
| CPU fine-tuning | Completed | 10 YOLO11 Nano epochs, real best.pt/last.pt, CSV and loss curves |
| MLflow training records | Completed | FINISHED run ab47990144a9475dab914f2c5dff5eaf, 10 epoch metric histories, verified best-model artifact |
| Held-out evaluation | Completed | Official Ultralytics comparison, per-class AP, PR/confusion plots and fixed-confidence IoU analysis |
| ONNX export/runtime | Completed | Valid opset-17 graph, FP32 CPUExecutionProvider, standalone Torch/Ultralytics import-block proof |
| Fair CPU benchmark | Completed | 3 isolated trials per engine, 300 measured forwards each, 10 warmups/trial; raw timings and memory retained |
| Existing tracking/logging | Completed | Original regression plus trained PyTorch/ONNX ByteTrack and ONNX BoT-SORT videos/CSVs decoded and checked |
| Dashboard integration | Verified programmatically | AppTest passes with original default, engine selection and measured-report section |
| Unit tests | Passed | 22 tests; the final dashboard/evaluation recheck also passed |
| Browser/camera checks | Not reverified | Tool-launched local servers were not reachable from the browser on this execution host; webcam not exercised |

## What was measured
Hardware: AMD Ryzen 7 260 w/ Radeon 780M Graphics, Windows 11, Python 3.12.14.
Training: 320px, batch 4, two CPU threads, AdamW, frozen first ten layers,
validation patience 4, AMP off, seed 42. Completed in 116.06 seconds
including fit/final validation/plot work, excluding dataset download and MLflow logging.

This is actual fine-tuning: 123
of 223 same-shaped parameter
tensors differ from the original. The original weight file hash is unchanged.
The selected checkpoint hash is 38e56ca53e8a6c50ec319c7b7b3e7ee854e18b102e7e13d82d1cb59112275c9d.

The fine-tuned model scored lower: mAP50:95 0.4629 versus
0.6859 pretrained. The original is retained as the dashboard
default. No hyperparameter/checkpoint tuning or additional training was performed
based on the test comparison.

Same fine-tuned model, FP32, batch 1, 320px, two threads:
- Mean forward latency: 36.801 ms PyTorch,
  20.032 ms ONNX Runtime.
- Latency ratio: 1.837x on this laptop.
- All 41 test detections matched at IoU >= .95. The independent comparison against
  official Ultralytics predictions also matched all 41.
- Separate-process peak RAM includes Python/libraries/loading/inputs; it is not
  model-only memory. ONNX benchmark workers never imported Torch.
- Video FPS is reported both with frame processing alone and with final H.264
  encoding time added. Setup, warmup and optional MLflow logging are separate.

The benchmark was repeated after functional tests completed, without those tests
running alongside the final timed trials. This is one machine and a small dataset,
not a general performance guarantee.

## Data qualifications
Known groups and exact duplicates do not cross splits. Source ID grouping is
inferred; verified camera/site/session metadata is not supplied by the source.
Residual unrecognized scene overlap cannot be ruled out. Thirty-six contact-sheet
images were visually reviewed; automated checks cover all 120 images/234 boxes.
Source human annotations include cropped people/body portions. The dataset is
positive-only and small. COCO pretraining overlap was not exhaustively audited.
See datasets/safevision/QUALITY_REPORT.md and LICENSE_AND_SOURCE.md.

## Integration and limitations
Direct inference_onnx.py works with Torch and Ultralytics imports deliberately
blocked. The tracker-integrated ONNX path imports Ultralytics tracking libraries
(which import torch), while model execution remains ONNX Runtime CPU.
Both ByteTrack and BoT-SORT remain available; appearance ReID is disabled.

A local MLflow file store and portable experiment export are supplied.
Live mlruns is omitted from ZIPs because artifact URIs contain absolute paths.
Use log_training.py to replay the completed training history after relocation.
It is explicitly tagged as replayed output; it does not train again.

Earlier work verified Streamlit in a browser. This upgrade is verified using
Streamlit AppTest; no new browser screenshot is claimed. Existing screenshots
are historical. Some headless test exits emitted a pyreadline3 console-cleanup
warning after assertions passed; the test process returned exit code zero.

The CPU-only ONNX additions were isolated from the original core dependencies.
All 117 installed distributions passed dependency checks.

## Evidence
- reports/upgrade_validation.json: checkpoint parameters, dependencies, video/CSV checks.
- reports/mlflow_training_run.json: portable params/epoch histories/status.
- reports/official_prediction_parity.json and torch_free_proof.json.
- reports/onnx_verification.json: graph, provider, shape and hashes.
- reports/evaluation_comparison.json and benchmark_results.json.
- Raw trials under reports/benchmark and original training history under training/cpu_person.

## Preserved versions
The project before this training upgrade remains in
SafeVision_AI_before_training_2026-10-25.zip alongside the new archive.
The earlier MLflow and pre-MLflow ZIPs are also retained.
