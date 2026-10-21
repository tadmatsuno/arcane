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

## Fixed: MOOG utilities require optional atmosphere installations at import

- Importing `moog` eagerly imported MARCS and AVG3D, which immediately load
  external grids and a user configuration file. This prevented output parsing
  and species conversion in a fresh environment.
- Load the requested atmosphere module only when generating or reading an
  atmosphere through that module. Missing MOOG configuration now follows the
  existing unconfigured-path behavior. Invalid JSON still raises an error.
- Validation: `test/test_moog_imports.py` imports in a fresh subprocess with a
  nonexistent configuration and verifies species conversion without loading
  either atmosphere module.

## Scope

This audit follows `Agents.md` and preserves the public APIs. The feature ideas
in `TODO` (batch configuration, detection limits, correction grids, and additional
synthesis engines) remain future work; they are not implemented by these fixes.
