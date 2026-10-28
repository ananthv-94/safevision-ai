# Interview guide — executed CPU workflow

## Architecture and problem
SafeVision detects people, tracks trajectories and flags entry into a configured
polygon. Training flow: human annotations → grouped splits → YOLO11 Nano CPU
fine-tuning → validation checkpoint selection → held-out test → ONNX export.
Runtime flow: OpenCV → PyTorch or direct ONNX detector → ByteTrack/BoT-SORT →
footpoint rule → debounced incident state → annotated MP4/CSV/Streamlit.
MLflow explicitly records completed inference and training outputs in one local store.

## Dataset and leakage
The SH17 subset has 120 workplace-related images and 234 person annotations:
84 training, 18 validation, 18 test. It retains existing human class-0 labels,
not detector-generated labels. Source URLs and hashes are included.
CC BY-NC-SA licensing makes this a noncommercial educational/interview experiment.
Original photograph terms remain applicable.

Known related photo-ID groups and exact/perceptual duplicate checks keep images
together. Groups are inferred because official camera/session metadata is absent.
Do not claim a guaranteed site-separated split. Residual scene overlap and source
annotation errors remain possible. There are no negative-only images, and some
labels enclose heavily cropped people. No exhaustive COCO pretraining overlap
audit was performed.

Training updates parameters; validation selects the checkpoint and settings;
test evaluates the frozen model. I did not retrain based on this test result.
Production needs independent site/camera/time holdouts and verified labeling policy.

## Why Nano and what fine-tuning changes
YOLO11 Nano is small enough for this CPU laptop. Pretraining supplies general visual
features; fine-tuning adjusts trainable detector layers for person labels.
I froze the first ten layers and used 320px, batch 4, two CPU threads, AdamW
learning rate .001, mild augmentations and validation-based early stopping.
Ten epochs actually completed in about 116 seconds. Of 223 comparable parameter
tensors, 123 changed. The original pretrained file remains unchanged.

Loss measures optimization fit, not accuracy. A decreasing training loss does not
prove better held-out performance. The included plots show box, classification
and distribution-focal training/validation losses. MLflow stores epoch losses,
validation metrics, dataset version/sizes, hyperparameters and the best checkpoint.

## Accuracy finding
Pretrained mAP50:95 was 0.686, while fine-tuned was 0.463.
I kept the original as the app default. Small data, a reset class head, limited
optimization and cropped-person label definitions are plausible explanations for
regression; this run does not prove which caused it. I would collect better data
and tune on validation, then use a fresh test set.

IoU is intersection area divided by union area. Precision asks how many detections
are correct; recall asks how many labeled objects were found. F1 balances them.
AP summarizes the precision-recall curve. mAP50 matches at IoU .50; mAP50:95
averages .50 through .95. With one class, mean AP equals person AP.
The report separates AP matching thresholds, NMS threshold, maximum-F1 P/R
summaries and a fixed-confidence IoU check. Confusion matrices have separate defaults.

## ONNX Runtime and measurement
ONNX represents the graph; ONNX Runtime executes CPU kernels. The direct module
implements square letterboxing, RGB FP32 /255, NCHW, YOLO11 output decoding,
numpy NMS and coordinate reversal. It succeeded with Torch and Ultralytics imports
blocked. The tracking integration imports tracker libraries that use torch, but
ONNX model execution is direct CPUExecutionProvider.

Both runtimes execute the same fine-tuned model at FP32, batch 1, 320px and two CPU
threads, using identical images/preprocessing/NMS. Three isolated trials per engine
use ten warmups and 100 timed forwards. Loading/warmup are outside steady-state timing.
Mean latency: 36.80 ms PyTorch vs 20.03 ms ONNX Runtime
(1.84x measured ratio). P95: 44.06 vs 25.24 ms.
Peak process RAM: 303.1 vs 146.9 MiB.
Whole-video FPS: 18.99 vs 27.40.
All 41 test detections matched at IoU >= .95, including the separate official
Ultralytics prediction comparison.

Graph fusion, kernels and runtime overhead are possible explanations, not proven
causal contributions. Model latency is distinct from decoding/tracking/drawing/writes.
Process RAM includes Python/libraries/model/inputs. PT stores half-precision weights
on disk while both runtimes execute FP32, so the FP32 ONNX file can be larger.
No universal ONNX speed claim is made.

## Tracking and safety logic
ByteTrack matches stronger detections and then weaker candidates for recovery.
BoT-SORT adds camera motion compensation; appearance ReID is disabled.
IDs are trajectory hypotheses, not verified identities. The bottom-center
footpoint is checked against a convex polygon, including its boundary.
Continuous occupancy logs one event; observed-exit confirmation and re-entry
cooldown reduce repetition. Fragmented IDs can still miss or duplicate alerts.

## Next engineering steps
Use independent workplace recordings, negative scenes and human-reviewed incident
labels. Measure missed entries and false alarms. Add durable storage, retention,
access controls, camera reconnects and bounded processing queues. Profile CPU
threads/resolution and validate accuracy after any further optimization.

## 2–3 minute presentation script
“I extended SafeVision from a working video demo into an end-to-end CPU learning
workflow. It detects people, tracks IDs and records entries into a staged restricted
zone, with an annotated video and incident CSV.

I used YOLO11 Nano because this laptop has no NVIDIA GPU. I prepared 120 workplace
images with existing human person labels and kept known related photos together
across training, validation and test splits. I checked annotation formats, image
integrity and known group/duplicate overlap, while documenting missing camera
metadata and source annotation limitations.

I actually trained for ten CPU epochs. Validation selected the best checkpoint;
test data stayed separate from training decisions. MLflow records the dataset
version, parameters, losses, validation metrics and model artifacts.

The important accuracy result was a regression: mAP50:95 fell from about .686 to
.463. I kept the stronger original model as the default. A completed training run
is not automatically a deployment improvement.

I exported that same trained model to ONNX and implemented direct CPU inference.
It works with Torch imports blocked. In controlled batch-one FP32 trials, average
model latency was about 36.8 milliseconds in PyTorch and
20.0 in ONNX Runtime, about 1.84 times faster on this laptop.
Predictions closely matched, and I separately measured full-video FPS and process RAM.

Both trackers, zone alerts and incident logs remain working. My next step would
be stronger independent workplace data and site-specific false-alarm/miss review.
The project demonstrates reproducible measurement and an evidence-based decision
to retain the baseline, rather than claiming improvements that were not measured.”
