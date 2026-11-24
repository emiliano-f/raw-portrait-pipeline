# Colour pipeline

## Working space

Between `decode` and `export` the image is **scene-linear float32 RGB**,
shape `(H, W, 3)`, with **sRGB / Rec.709 primaries and a D65 white point**.
Linear means pixel values are proportional to scene light: doubling the light
doubles the value. No transfer function (gamma) is applied in the working
space.

- The `rawpy` decoder asks LibRaw for 16-bit linear output (`gamma=(1, 1)`,
  `no_auto_bright=True`, `output_color=sRGB`, camera white balance) and
  divides by 65535. With `adjust_maximum_thr=0` LibRaw never lowers the white
  level per image, so **1.0 is the sensor white level** in every file
  ([details](components/decode-io.md#rawpy-decoder)).
- Values above 1.0 are allowed and kept (exposure gain, highlight headroom).
  Nothing clips before the exporter.
- Luminance is `Y = 0.2126 R + 0.7152 G + 0.0722 B` (Rec.709),
  `raw_pipeline.core.color.luminance`. Metering, tone and QA all use it.

## Where sRGB encoding happens

The sRGB transfer function is defined once, in
`raw_pipeline.core.color.srgb_encode` (IEC 61966-2-1, clips to `[0, 1]`
first):

```text
V = 12.92 * L                    if L <= 0.0031308
V = 1.055 * L ** (1 / 2.4) - 0.055   otherwise
```

It is applied only where pixels leave the working space:

| Where | What | Module |
| --- | --- | --- |
| Exporters (`jpeg`, `tiff`) | final images, with the sRGB ICC profile | `raw_pipeline.exporters.encoding` |
| `report` stage | contact-sheet thumbnails | `raw_pipeline.exporters.encoding` |
| `meter` stage | the detection preview only; metering reads the linear image | `raw_pipeline.stages.meter` |
| `qa` stage | the downscaled preview sent to the LLM | `raw_pipeline.qa.preview` |

Resizing happens **before** encoding (in linear light, OpenCV `INTER_AREA`),
so downscaled edges and highlights stay photometrically correct. Then values
are quantised with `to_uint8` / `to_uint16` (round half to even). No stage
between `decode` and `export` may encode or clip.

## Exposure (EV) maths

One EV (stop) is a factor of 2 in light. Every exposure change is a
multiplication in linear space:

```text
image' = image * 2 ** EV
```

never an addition and never on encoded values.

The `meter` stage measures a luminance `m` (faces: trimmed mean of the central
skin area; no face: centre-weighted median) and computes

```text
EV = clamp(log2(t / max(m, 1e-6)), -E_max, +E_max)
```

with target `t = meter.target_face_luminance` (or `target_center_luminance`)
and `E_max = meter.max_correction_ev`. The `tone` stage applies `2**EV`, then
a luminance curve that is the identity between the shadow pivot and the
highlight shoulder start (with the defaults, `0.05 ≤ Y ≤ 2**-1.5 ≈ 0.35`), so a
face metered to 0.18 lands exactly on 0.18. Highlights up to
`2**tone.highlight_rolloff_ev` (default 4 EV above white) are rolled into
1.0 instead of clipping; the curve multiplies RGB by `curve(Y)/Y`, so hue and
saturation are kept. Formulas: [Tone stage](components/tone-retouch.md#tone-stage).

Option names carry their unit: `_ev` is stops, `_luminance` is linear `Y`.
