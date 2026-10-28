# CPU training guide

## What was executed

The included run trained YOLO11 Nano for 10 epochs on CPU with 84 training
images. The 18 validation images selected the best checkpoint; the 18 test images
were held out until final comparison. The trained checkpoint has one output class,
person. The original COCO model is retained unchanged as the application default.

The new detector did not outperform the original on the small held-out set.
This is a legitimate small learning experiment, not a claim of production
improvement. The evaluation report records the regression.

## Data and annotations

SH17 is the authors' industrial/workplace safety dataset:
https://github.com/ahmadmughees/SH17dataset .
Annotations are CC BY-NC-SA 4.0; photographs retain Pexels terms. Credit the authors,
Hafiz Mughees Ahmad and Afshin Rahimi, and use this subset for noncommercial study.

The preparation script uses a pinned source URL manifest and original human
YOLO annotations from the public Kaggle source. It retains class 0, clips
coordinates to the image, and downloads/resizes uncropped photos. It does not
ask a detector to create ground truth. A manifest records source URLs, original
label hashes, final image hashes, boxes and split/group assignments.

The 120-image subset covers construction, workshops, offices and service work.
It contains 234 person instances. It is positive-only; it cannot establish
background-only false-alarm rates. Some original annotations contain heavily
cropped people/body portions. This policy and the small sample limit its value
as a deployment evaluation dataset.

Related source photo IDs are grouped conservatively, then perceptual near
duplicates merge groups. No known groups or exact duplicates cross splits.
These are inferred shooting groups, not official camera/session IDs, which are
not available. A stronger production evaluation needs verified camera/site/time
groups and independent workplace recordings. No guarantee is made that every
visually related scene has been discovered.

Automated checks cover every image/label and group/hash overlap. Contact sheets
show all converted annotations; selected sheets were visually inspected.
The source annotations are treated as reference labels, not perfect truth.
COCO pretraining/source overlap was not independently exhaustively audited.

## Reproduce

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --count 120 --seed 42
.\.venv\Scripts\python.exe train.py --epochs 10 --imgsz 320 --batch 4 --threads 2 --output training/my_cpu_run
```

Prepared images are included, so downloading is optional. If source URLs change
or downloads fail, use the bundled dataset and preserve its manifest.
The script recreates its own dataset image/label folders. Do not add unrelated
files to those generated folders.

## Settings and resource control

- Model: pretrained yolo11n.pt, with a single-class detection head.
- Input: 320 pixels. Batch: 4. CPU threads: 2. Data-loader workers: 0.
- AdamW, learning rate .001, final learning-rate fraction .1.
- Freeze first 10 model layers; train the remaining detector layers.
- Up to 10 epochs, patience 4 for validation-based early stopping.
- AMP disabled. Seed 42 and deterministic mode enabled.
- Horizontal flips, mild rotation/translation/scale, mosaic .2, no mixup.
- The final two epochs close mosaic. All settings are saved in args.yaml.

On a slower laptop, begin with batch 2 or fewer epochs and use a new output
directory. Changes based on validation data are legitimate; do not select settings
from the test score. Larger input sizes may recover small workers but cost CPU time.

## Outputs and loss interpretation

`results.csv` records per-epoch box, classification and distribution-focal losses
for training and validation, plus validation P/R/mAP and learning rates.
`loss_curves.png` plots the actual loss history. `weights/best.pt` is selected by
Ultralytics validation fitness; `weights/last.pt` is the final epoch.
The selected model is copied to `models/best.pt`.

Training loss measures fit to augmented training examples. A decreasing loss does
not prove improved held-out accuracy. Validation loss/metrics can oscillate on
18 images. Small data, resetting a class head and limited optimization can reduce
a pretrained detector's quality, as this experiment demonstrates.

MLflow's **SafeVision training** experiment records the dataset hash/size,
architecture, hyperparameters, per-epoch losses/validation metrics, plots and best
checkpoint. The existing inference experiment remains separate. This is explicit
experiment tracking, not global autologging or model-registry deployment.
If logging fails, inspect `mlflow_training.json`; local training outputs remain.

## After training

```powershell
.\.venv\Scripts\python.exe evaluate.py --compare
.\.venv\Scripts\python.exe export_onnx.py
.\.venv\Scripts\python.exe benchmark.py --video demo/warehouse.mp4 --video-frames 100
```

Checkpoint hashes link the training result, exported model and benchmark. Never
rename the original pretrained file to best.pt and call that training.
