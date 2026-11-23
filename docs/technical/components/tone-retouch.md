# Tone and retouch

Package 4 of the [initial build plan](../../plan/initial-build.md). Owns
`stages/tone.py`, `stages/retouch.py` and `stages/retouchers/`.

| Registry | Name | Class |
|---|---|---|
| `stages` | `tone` | `ToneStage` |
| `stages` | `retouch` | `RetouchStage` |
| `retouchers` | `none` | `NullRetoucher` (Null Object) |
| `retouchers` | `frequency_separation` | `FrequencySeparationRetoucher` |

Importing `raw_pipeline.stages.tone`, `raw_pipeline.stages.retouch` and
`raw_pipeline.stages.retouchers` registers all four.

## Tone stage

Input and output are scene-linear working-space RGB; nothing is clipped.

1. **Exposure**: `image * 2**correction_ev` from the `EXPOSURE` artifact (0 EV when absent).
2. **Curve on luminance**: `Y` = Rec.709 luminance (`core.color.luminance`); every
   pixel is multiplied by `curve(Y) / Y`, which keeps RGB ratios (hue and
   saturation). At `Y <= 0` the gain is its limit, `2**shadow_lift_ev`.

`ToneCurve` is `shoulder(lift(Y))`:

| Part | Formula | Range |
|---|---|---|
| Shadow lift | `Y * 2**(shadow_lift_ev * (1 - Y/p)**2)` | `Y < p = shadow_pivot_luminance` |
| Identity | `Y` | `p <= Y <= s` |
| Highlight shoulder | `s + d / (1 + d/k)`, `d = Y - s` | `Y > s = 2**highlight_rolloff_start_ev` |

`k = (W - s)(1 - s) / (W - 1)` with `W = 2**highlight_rolloff_ev`, so scene
luminance `W` (that much headroom above diffuse white) lands exactly on 1.0;
brighter values keep rising towards `s + k` (> 1), so the exporter does the
only clip. Both knees have slope 1, so the curve is C1. It is strictly
monotonic while `shadow_lift_ev < 2 / ln 2 ≈ 2.885 EV`; the stage rejects larger
values, `highlight_rolloff_start_ev >= 0` and non-positive pivots with a
`ConfigError` when it is built. Knees are evaluated in float64 on the affected
pixels only, so float32 rounding cannot break monotonicity.

With the defaults (`-1 EV`, `1.5 EV`, `0.3 EV`, `0.05`) the curve is the
identity between luminance 0.05 and 0.5, so exposure scaling is exact for the
face (metered to 0.18).

Recorded as `tone.*`: `exposure_ev`, `exposure_method` (`none` without
measurement), the four curve parameters and `clipped_fraction` (fraction of
pixels with a channel > 1.0 after tone, i.e. clipped at export).

## Retouch stage

`RetouchStage` passes `ctx.image` and the `FACES` artifact (empty when absent)
to the retoucher named by `retouch.method` and records `retouch.method` plus the
retoucher's parameters. `none` returns the same array.

### `frequency_separation`

Per face, in a region around the face box padded by three smoothing sigmas:

1. **Skin mask** (`SkinMaskBuilder`):
    - With a MediaPipe 478-point mesh (`landmark_scheme == "mediapipe_face_mesh_478"`):
      convex hull of the face oval minus the hulls of each eye (with iris),
      eyebrow, the lips (covers the mouth interior and teeth) and the
      nose tip/alae (nostrils). Protected hulls are dilated by 3 % of the face
      box width. Landmark indices come from MediaPipe's `FaceLandmarksConnections`.
    - Otherwise (OpenCV fallback, other schemes): an ellipse inscribed in the
      box minus an eye/brow band across the box (15-55 % of the box height), a nose
      box and a mouth box. Generous on purpose: detector boxes vary.
    - The protected features of every *other* face in the image are removed
      too, so overlapping faces (group shots) never smooth each other's eyes or lips.
    - The hard mask is feathered *inwards* over `mask_feather_px` (smoothstep of
      the distance to the mask edge), so the soft mask is exactly 0 outside the
      face and on every protected feature.
2. **Split**: `L = blur(image, σ)` with `σ = blur_radius_face_fraction × face box
   width` (≥ 0.5 px); texture `image − L` is never modified.
3. **Smooth the low-frequency layer**: `L' = blur(L, 3σ)`. Both blurs are
   normalised by the hard skin mask (only skin pixels contribute), so hair,
   eyes or background do not bleed into the skin.
4. **Blend**: `image + strength × soft_mask × (L' − L)`; pixels with mask 0 are
   copied bit-for-bit.

Only tone blotches between σ and 3σ are reduced, by at most `strength`; pores
and fine texture stay. No geometry changes, no generative model, deterministic
(OpenCV Gaussian blur and distance transform).

Recorded as `retouch.*`: `method`, `strength` (0 when nothing was retouched),
`mask_coverage` (fraction of image pixels with mask > 0), `faces_retouched`,
`mask_source` (`landmarks`, `ellipse`, `mixed` or `none`), `blur_radius_px`
(largest σ) and `mask_feather_px`.

## Tests

`tests/unit/stages/`: both stages pass `StageContract` (tone with and without
`EXPOSURE`; retouch with `none` and `frequency_separation` on a synthetic face).
Tone: monotonic and finite on a ramp, identity in the mid-tones, exact EV
scaling, C1 at both knees, scene white → 1.0, values above 1 preserved, hue
ratios kept, invalid parameters rejected. Retouch: synthetic 478-point faces
with known feature ellipses (`tests/unit/stages/retouchers/faces.py`); bitwise
unchanged outside the mask and on eyes/brows/lips/nose, blotch band reduced,
texture band std ≥ 95 %, effect linear in strength, passthrough at strength 0
or without faces, ellipse fallback, faces cut by the image border.

## API

::: raw_pipeline.stages.tone
::: raw_pipeline.stages.retouch
::: raw_pipeline.stages.retouchers.frequency_separation
::: raw_pipeline.stages.retouchers.skin_mask
::: raw_pipeline.stages.retouchers.null
