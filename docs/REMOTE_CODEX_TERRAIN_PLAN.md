# Remote Codex Plan For Robust Terrain-Aware Learned Motion Matching

## Goal

Make the learned controller reproduce the terrain-specific motions that are visible in ordinary motion matching, while preserving learned playback smoothness and eliminating catastrophic root explosions.

This plan is intentionally biased toward robustness. Do not spend time on more runtime smoothing hacks until the data and training issues below are fixed.

## Current Diagnosis

1. Ordinary motion matching can surface terrain-specific uphill/downhill/rough-ground clips because it directly selects real database frames.
2. Learned motion matching is smoother because it projects onto and rolls along a learned feature manifold.
3. The learned stack currently washes out rare terrain-specific behaviors because:
   - the environment signal is too sparse;
   - bad frames exist near some clip boundaries;
   - projector training targets nearest features, which biases toward common smooth modes;
   - stepper training favors stable low-curvature evolution of features and latent state;
   - decompressor training does not explicitly enforce terrain-contact geometry.

## Hard Requirements

1. Do not rely on controller-only fixes as the main solution.
2. Treat bad database frames as a training-data bug, not a gameplay-tuning problem.
3. Every new learned model must be evaluated against the ordinary motion-matching baseline on the same terrain scenes.
4. Every stage must leave artifacts on disk so failures are inspectable.

## Phase 1: Data Sanitation

### 1.1 Build a frame-quality audit

Implement a standalone Python audit script under `resources/` that reads:

- `database.bin`
- `features.bin`
- `terrain_features.bin`

The script must produce:

- per-frame root planar speed;
- per-frame root angular speed;
- per-range summary stats;
- a binary frame-valid mask;
- a CSV report of rejected frames with reason codes.

Reject at minimum:

- frames with root planar speed outside a robust quantile-based limit;
- frames in the first `N` and last `N` frames of each range;
- frames with duplicated or zeroed root positions followed by extreme root velocities;
- frames with obviously discontinuous root transforms relative to neighboring frames.

Write outputs:

- `frame_mask.bin`
- `frame_audit.csv`
- `frame_stats.json`

### 1.2 Rebuild range-safe training windows

Update all training scripts so sampled windows never cross invalid frames.

Do not merely skip bad samples inside the loop. Build valid contiguous training spans first, then sample only from those spans.

Acceptance:

- no sampled decompressor/stepper/projector training window may contain an invalid frame;
- no search/runtime candidate frame should come from the invalid mask.

## Phase 2: Environment Representation Upgrade

### 2.1 Increase lookahead density

Replace the current three-horizon environment layout with five horizons:

- `10`
- `20`
- `40`
- `60`
- `80`

Keep right/center/left sampling at each horizon.

Environment feature target:

- terrain height: `5 * 3 = 15`
- obstacle SDF: `5 * 3 = 15`
- total environment features: `30`
- total matching feature size: `27 + 30 = 57`

### 2.2 Keep offline and runtime sampling identical

The same layout, strip width, coordinate frame, and height baseline must be shared between:

- PFNN export
- terrain/environment asset generation
- runtime query construction
- debugging visualization

There must be one source of truth for:

- horizon list
- strip width
- SDF clamp distance

## Phase 3: Training Objective Upgrade

### 3.1 Projector

Keep nearest-neighbor supervision, but add penalties that stop projector outputs from collapsing rare terrain actions back to smooth average states.

Add:

- higher loss weight on environment feature reconstruction;
- feature-group weights so terrain/SDF errors matter at least as much as trajectory direction errors;
- an optional hard-mining batch mode that oversamples large-environment-gradient frames.

Acceptance:

- projector error on environment dimensions must be reported separately;
- rare terrain windows must not regress toward flat-ground neighbors by default.

### 3.2 Stepper

Current stepper only rolls `(X, Z)` forward. Make it condition on the current query or environment control so terrain changes are injected every step rather than only through occasional reprojection.

Recommended direction:

- input: `[X_t, Z_t, Q_t]`
- output: `[dX_t, dZ_t]`

Where `Q_t` contains at least:

- trajectory positions;
- trajectory directions;
- environment features.

If full query conditioning is too large, at minimum condition on the environment slice and future trajectory slice.

Acceptance:

- stepper rollout should track terrain transitions without requiring aggressive projector correction every frame.

### 3.3 Decompressor

Add terrain-contact losses instead of relying only on pose reconstruction.

Add losses for:

- support foot height relative to sampled ground;
- contact foot vertical velocity when contact is active;
- penetration penalty for feet below terrain;
- floating penalty for contact feet too far above terrain.

These losses must be computed in world space using the same terrain field as runtime.

Acceptance:

- on validation clips with clear slope changes, contact feet should remain near terrain during support.

## Phase 4: Evaluation Harness

Build an automated comparison harness that runs both:

- ordinary motion matching
- learned motion matching

on the same scripted controller inputs and terrain scenes.

Metrics to log:

- root trajectory error relative to selected database baseline;
- contact foot terrain height error;
- contact foot slip;
- terrain penetration count;
- projector correction magnitude over time;
- stepper drift over time.

Required evaluation scenes:

1. Flat ground.
2. Sustained uphill.
3. Sustained downhill.
4. Rough rocky field.
5. Obstacle avoidance through boxes.

Write results to:

- `eval_metrics.json`
- `eval_summary.md`
- optional plots under `eval_plots/`

## Phase 5: Runtime Integration

Only after retraining:

1. regenerate `terrain_features.bin`;
2. rebuild `features.bin`;
3. retrain `decompressor.bin`;
4. retrain `latent.bin`;
5. retrain `stepper.bin`;
6. retrain `projector.bin`;
7. run the evaluation harness;
8. switch runtime defaults to the best validated model.

## What Not To Do

1. Do not add more ad hoc root smoothing before fixing the training set and objectives.
2. Do not increase projector blending to hide bad terrain behavior.
3. Do not judge terrain quality from query debug spheres alone.
4. Do not trust any model trained on the current dirty frame set.

## Deliverables Expected From Remote Codex

1. Code changes for all phases above.
2. A reproducible command sequence for data rebuild and training.
3. Saved audit artifacts and evaluation artifacts.
4. A short report answering:
   - which bad-frame categories were found;
   - whether learned terrain-contact metrics improved over baseline;
   - whether learned terrain-specific motions now visibly match ordinary motion matching on slopes and rough ground.

## Acceptance Criteria

The work is complete only if all of the following hold:

1. Ordinary motion matching no longer explodes because bad frames are excluded from both search and sequential playback.
2. Learned motion matching remains smooth.
3. Learned motion matching visibly exhibits uphill/downhill/rough-ground behaviors that are present in ordinary motion matching.
4. Contact feet on slopes and rough terrain stay substantially closer to the ground than with the current model.
5. The full pipeline is reproducible on the remote SSH server from raw data to final artifacts.
