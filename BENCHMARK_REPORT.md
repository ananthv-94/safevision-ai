# CPU benchmark

| Metric | PyTorch CPU | ONNX Runtime CPU |
|---|---:|---:|
| average_latency_ms | 36.801 | 20.032 |
| median_latency_ms | 35.640 | 19.163 |
| p95_latency_ms | 44.056 | 25.237 |
| inference_fps | 27.173 | 49.920 |
| detection_pipeline_fps | 25.987 | 46.068 |
| preprocess_ms | 1.258 | 1.256 |
| postprocess_ms | 0.421 | 0.419 |
| peak_process_ram_mb | 303.090 | 146.945 |
| model_mb | 5.171 | 9.955 |
| video_processing_fps | 18.991 | 27.397 |

Measured native/ONNX latency ratio: **1.837x**. Greater than 1 means ONNX is faster.

FP32, batch 1, 320px, 2 CPU threads; 3 isolated trials per engine, 100 measured forwards and 10 warmups each. Engine order alternates between trials. Same held-out images and shared preprocessing/NMS.

Model forward excludes loading, warmup, preprocessing and NMS. Detection pipeline adds preprocessing/postprocessing, excludes disk reads. Separate process high-water RSS/Windows peak working set; includes Python, libraries, model, inputs and loading; not model-only memory.

Prediction consistency: {"torch_detections": 41, "onnx_detections": 41, "matched_at_iou95": 41, "matched_fraction": 1.0, "mean_matched_iou": 0.9999995512456792, "maximum_confidence_error": 9.238719940185547e-07, "matching_threshold": 0.95, "confidence": 0.25, "nms_iou": 0.7}.

The native checkpoint stores FP16 weights on disk but both engines execute FP32. The ONNX graph stores FP32 weights; file size is not memory footprint.

Differences can arise from graph fusion, kernel implementations and runtime overhead. No causal performance attribution is established by this single-laptop experiment. Full video throughput also includes tracking, drawing and encoding writes, so model speedup need not equal video speedup. The ONNX benchmark workers assert that PyTorch was never imported. The tracker-integrated app uses Ultralytics tracking libraries which may import torch, but ONNX model execution is always direct CPUExecutionProvider.

## Final-file encoding included

The additional video_with_final_encoding_fps metric includes both frame processing and the final H.264 pass, while excluding setup/warmup and optional MLflow logging. It is computed from measured processing_seconds + encoding_seconds.

- pytorch: 16.317 FPS.

- onnxruntime: 22.362 FPS.
