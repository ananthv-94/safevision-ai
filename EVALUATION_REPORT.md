# Held-out evaluation

| Metric | Pretrained | Fine-tuned |
|---|---:|---:|
| precision | 0.8683 | 0.7319 |
| recall | 0.8597 | 0.7123 |
| f1 | 0.8640 | 0.7220 |
| map50 | 0.8849 | 0.7352 |
| map50_95 | 0.6859 | 0.4629 |

Fine-tuning mAP@50:95 change: -0.2230.
This is an 18-image grouped test subset, not a production accuracy estimate. No test data was used for checkpoint selection. Precision/recall/F1 are at the same Ultralytics validation operating point. NMS IoU is .70; AP matching spans .50:.95, candidate confidence .001. Single class: person.

A small CPU experiment can improve or degrade a pretrained model. Small data, cropped-person annotation policy, head re-initialization and limited optimization are plausible causes; they are not proven by this run. Do not tune on this test result. Collect independent site data before deployment.

## Fixed-confidence IoU analysis

| Model | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained | 39 | 7 | 7 | 0.8478 | 0.8478 | 0.8478 | 0.8744 |
| fine_tuned | 33 | 8 | 13 | 0.8049 | 0.7174 | 0.7586 | 0.8304 |

Fixed analysis: confidence .25, match IoU .50, NMS IoU .70, common square letterboxing. Main P/R/F1 above are Ultralytics maximum-F1 interpolated summaries, at confidence .3203 pretrained and .2823 fine-tuned; these are descriptive test-curve points, not deployment thresholds tuned on validation. Confusion-matrix plots use confidence .25 and match IoU .45.

**Decision:** retain the stronger pretrained default. Do not promote this small fine-tune as an accuracy improvement. No test-driven retraining was performed.
