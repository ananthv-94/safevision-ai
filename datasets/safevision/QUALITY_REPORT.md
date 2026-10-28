# Dataset quality

| Split | Images | Person instances | Inferred groups |
|---|---:|---:|---:|
| test | 18 | 46 | 5 |
| val | 18 | 39 | 5 |
| train | 84 | 149 | 23 |

All 120 images decode and all 234 boxes pass shape, class and bounds checks. No missing labels, corrupt final images, exact duplicates or known cross-split group overlap. Source: original SH17 human person annotations. No pseudo-labels.

Photo-ID adjacency is an inferred shooting-group proxy, not official camera metadata. pHash distance <=8 merging found no extra near duplicates. Residual scene overlap is possible; this is not a site-separated production benchmark.

36 images on contact sheets 00, 02 and 04 were visually reviewed. Boxes align with visible people; heavily cropped people/body portions appear in the source label policy. Source annotations are not guaranteed perfect.

One class means inter-class imbalance is inapplicable. All images are person-positive, limiting false-alarm evaluation. COCO pretraining overlap was not independently exhaustively audited.

See LICENSE_AND_SOURCE.md and manifest.json for source URLs, label/image hashes, licensing and changes. Dataset version: 0bf028eba59390ea83a3d79aa443040c8dc94fe963058fd8413aee3d3412f3c5
