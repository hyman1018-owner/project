# Master Feature Automation Standards

## Model policy

- Detection / tracking default model: `yolo26s.pt`.
- Identity ReID default model: `yolo26s-reid.onnx`.
- Do not mix ReID embeddings from different ReID models in the same Master gallery.
- Future automation should write the ReID model name into each feature metadata row and compare only same-model features.

## Official feature source

- `master` is the only official ReID identity source.
- `pending` and `candidate` are staging concepts and must not be used as final identity proof unless promoted.
- A feature enters `master` only after crop quality, duplicate, and identity consistency checks.

## Minimum crop quality

Use the current runtime checks as the first gate:

- Crop width >= 32 px.
- Crop height >= 32 px.
- Crop area >= 1600 px.
- Brightness between 20 and 240.
- Sharpness >= 5.

Recommended automation threshold for stronger quality:

- Prefer crop width >= 80 px.
- Prefer crop height >= 60 px.
- Prefer YOLO detection confidence >= 0.75.
- Prefer quality score >= 0.80.
- Reject heavy occlusion, partial vehicle crops, motion blur, strong reflections, and crops dominated by background.

## Duplicate and diversity policy

- Current duplicate rejection threshold: cosine similarity >= 0.985 against same-GID same-gallery features.
- Automation should avoid adding near-identical consecutive frames.
- Prefer diversity over raw count:
  - front
  - rear
  - left side
  - right side
  - front-left / front-right
  - rear-left / rear-right
  - near and far scale
  - different lighting zones

## Per-GID target distribution

Minimum practical baseline:

- Single camera fixed angle: 8-12 high-quality Master features.
- Multi-angle same venue: 20-50 high-quality Master features.
- Cross-camera or broadcast-like video: 50-150 high-quality Master features.

Current hard cap:

- 500 Master features per GID.

Do not fill 500 with consecutive near-duplicates. A smaller gallery with diverse viewpoints is better than a large gallery of the same angle.

## Tracklet sampling rule

For future automatic capture:

- Sample from stable tracklets, not isolated detections.
- Require the same local track to persist for several frames before sampling.
- Sample one feature every N frames or when angle / size / location changes meaningfully.
- Keep top-quality frames per tracklet.
- Avoid frames near scene cuts, sudden tracker ID switches, or low-confidence detections.

## Find GID matching rule

- Query crops come from current detections.
- Query crops must pass crop quality filter before ReID extraction.
- Query embedding is compared against selected GID's Master features only.
- Current acceptance threshold: 0.72.
- If top score is below threshold, remain searching.
- Future automation should log top score, second score, margin, feature id, ReID model, and crop quality for debugging.

## Recommended future UI/debug fields

- ReID model used per feature.
- Last Find GID top score.
- Last matched feature id.
- Feature angle label.
- Feature quality histogram.
- Per-GID feature count by model and tracklet.
