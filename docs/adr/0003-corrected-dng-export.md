# ADR 0003: Corrected RAW export by patching a converter-made DNG

- Status: accepted
- Date: 2026-11-25

## Context

Users want the defect line removed from a RAW file they can keep editing.
CR2 is a proprietary format: there is no maintained library that writes it, and
rewriting its lossless-JPEG raw data would risk files Canon's and third-party
software read differently. DNG is an open TIFF-based RAW format every major
editor reads, and free converters produce it from CR2.

Writing a DNG from scratch would mean re-creating colour matrices, camera
profiles, EXIF and maker data, which converters already get right.

## Decision

1. **Base DNG from a pluggable converter.** A `DngSource` (registry
   `dng_sources`) produces an uncompressed, mosaic (CFA), 16-bit DNG: `dnglab`
   (default, open source), `adobe_converter`, or `existing_dng` for files
   already converted. Required converter flags are added by the code.
2. **Correct with ADR 0002.** The DNG photosites under the CR2 visible area
   become a `CFAFrame` (DNG black/white levels) and the configured
   `DefectCorrector` corrects it with `defects.map`. No DNG-specific correction.
3. **Patch in place.** `DngRawWriter` copies the base and overwrites only the
   bytes of changed photosites. The TIFF structure is not rebuilt, so all other
   metadata survives. Compressed, linear, non-16-bit or linearised DNGs are
   rejected rather than re-encoded.
4. **Exact coordinate mapping.** The offset between CR2 visible coordinates and
   DNG stored coordinates is accepted only when all reference photosites match
   exactly with the same CFA phase; candidates come from `ActiveArea`,
   `DefaultCropOrigin` and a ±64 photosite search. No exact unique match: no
   output.
5. **Digests.** Recompute `RawImageDigest`/`NewRawImageDigest` only when the
   implementation reproduces the stored values from the base data; otherwise
   remove both tags. Never leave a stale digest.
6. **RawDataUniqueID.** Adobe apps identify raw data by this ID (cached
   renders, catalog matching), so a corrected DNG must not keep the base's ID.
   When photosites changed it is replaced in place (16 bytes, no structural
   change) with `MD5(old ID + NewRawImageDigest of the corrected raw data)`:
   deterministic, and different from the base and from other corrections.
   Passthrough output keeps it (byte-identical). The report records it.
7. **Previews.** `raw_export.preview`: `regenerate` (default; render from the
   corrected DNG and replace in place where the format allows), `keep`, or
   `remove`. The applied handling is reported per image.
8. **Verification by default.** After writing: LibRaw opens it, untouched
   photosites are byte-identical, corrected ones equal the corrector output,
   each configured defect's detector score dropped (a weaker residual next to
   a thin scene stripe passes with a warning), the layout is intact, inputs
   are unchanged. A failing output is never published.

## Consequences

- The output depends on the converter's DNG; DNGLab 0.8.0 was checked to keep
  every EOS 800D photosite value. Converters that change values (linearisation,
  compression) are rejected by the format checks or by the exact mapping.
- `existing_dng` needs the original CR2 next to the DNG to prove the mapping.
- A base DNG that embeds its original raw file (`OriginalRawFileData`) is
  rejected: the embedded CR2 still has the defect (and its own GPS).
  `raw_export.allow_embedded_original` accepts it with a warning, only with
  `keep_gps`.
- New dependencies: none (tifffile, rawpy, Pillow are already used). DNGLab and
  Adobe DNG Converter are external tools installed by the user (CI installs
  DNGLab for the integration test).
- `core/` gains the DNG value objects and the `DngSource`/`DngRawWriter`
  protocols; adapters stay in `io/dng/`, mapping in `defects/`, orchestration
  in `runtime/raw_export.py`.
