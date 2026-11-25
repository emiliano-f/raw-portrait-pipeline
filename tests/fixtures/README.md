# Test fixtures

- `dark/dark_iso400.CR2`: the only committed RAW. A lens-cap-on frame from the
  maintainer's Canon EOS 800D (Rebel T7i), ISO 400, 1/125 s, with the sensor's
  hot column at visible x = 1509 (and a weaker one at 1510). It contains no
  personal data: GPS, camera/lens serial numbers and owner fields were removed
  with ExifTool, and the raw photosites are unchanged. Defect detection and
  `fix-raw` integration tests use it, in CI too.
- Real portraits are personal data and are **never committed** (no Git LFS
  either). Keep them in a local folder and point `RAW_PIPELINE_FIXTURES_DIR` at
  it; tests that need them skip when it is unset. `*.CR2` is git-ignored.
