# Executed CPU training

Model: YOLO11 Nano, one class (person).

Epochs completed: 10. Duration including fit/final validation/plot work: 116.06 seconds.

Train/validation/test: 84/18/18 images. Best checkpoint selected by validation fitness, never test score. AdamW .001, 320px, batch 4, two threads, freeze 10, patience 4, workers 0, AMP off.

Actual checkpoint: models/best.pt. Loss curves, official args.yaml, results.csv and best/last weights are included. The model underperformed the original on held-out data; see EVALUATION_REPORT.md.

MLflow status: FINISHED. Portable record: reports/mlflow_training_run.json.
