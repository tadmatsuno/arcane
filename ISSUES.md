# Repository bug audit

## Fixed: line-profile parameter bookkeeping

- Shared FWHM fitting built duplicate parameter entries and truncated line indices.
  It could leave a line at its initial width or fail with an empty index when other
  lines had fixed widths. Build one shared parameter for the free lines.
- Adding or deleting lines left depth and per-line fitting controls out of sync.
  Update every per-line array together. Added lines inherit the last line's
  fitting controls (or the defaults when there are no lines).
- Validation: `test/test_line_profile_regressions.py` checks shared wavelength
  shifts and widths, fixed widths, and fitting after adding/removing lines.
  All three tests failed before the fix and pass afterward.

## Scope

This audit follows `Agents.md` and preserves the public APIs. The feature ideas
in `TODO` (batch configuration, detection limits, correction grids, and additional
synthesis engines) remain future work; they are not implemented by these fixes.
