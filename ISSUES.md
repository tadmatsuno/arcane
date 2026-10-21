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

## Fixed: MOOG summary parsing loses rows or hangs

- EW and abundance parsers stopped reading after each table's first row. The
  abundance parser could also stop at the first species' average. Keep reading
  through data rows and resume at subsequent species headers.
- Synthesis parsing looped forever at EOF if no `MODEL` header was written.
  Raise `ValueError` with the file path instead.
- Validation: `test/test_moog_summary.py` covers multiple EW rows, multiple
  abundance rows and species, valid synthesis output, and missing headers.
  Both multi-row tests reproduced lost rows before the fix.

## Fixed: depth grids are not guaranteed to be monotonic

- The inverse depth grid removed points only when shallower than their immediate
  neighbor. A sequence such as `0.1, 0.4, 0.2, 0.3, 0.5` still contained a decrease
  after filtering and crashed the interpolator. Retain only increasing maxima.
- Reject depth windows containing no grid wavelengths, rather than silently
  selecting a wavelength outside the window. Report a clear error when fewer
  than two positive increasing depth samples remain.
- Validation: `test/test_grid_regressions.py` reproduces the nonmonotonic crash,
  checks positive and negative depth interpolation, and checks invalid windows
  and insufficient depth samples.

## Scope

This audit follows `Agents.md` and preserves the public APIs. The feature ideas
in `TODO` (batch configuration, detection limits, correction grids, and additional
synthesis engines) remain future work; they are not implemented by these fixes.

## Fixed: Turbospectrum cannot use the AVG3D grid

- An integration probe reproduced `KeyError: 'avg3d'` despite accepting the
  `mdlatm_io` parameter. The wrapper also hardcoded MARCS retrieval and export.
- Select the requested atmosphere reader and grid. Export AVG3D through
  Turbospectrum's native Stagger depth/temperature/density format, reconstructing
  geometrical depth from the grid's Rosseland extinction. Turbospectrum then
  computes its own reference-wavelength optical depths and equation of state.
  This uses the averaged atmosphere, not a full 3D synthesis calculation.
- Propagate metallicity, alpha abundance, and geometry from opacity generation
  into synthesis. Saved AVG3D atmosphere files can also be used.
- Validation: real synthesis and abundance-response checks with both programs
  and both grids, saved AVG3D synthesis, and an analytic depth-export check in
  `test/test_synthesis_atmospheres.py`. Existing synthesis/model tests pass.

## Reproducing validation

Dependencies were installed with `uv pip install --python .venv/bin/python -r
requirements.txt` in a virtual environment created with
`uv venv --python /usr/bin/python3 .venv` (Python 3.12).

Run the regression tests and existing VALD/species-conversion tests with:

```sh
PYTHONPATH=test MPLCONFIGDIR=/tmp/arcane-mpl .venv/bin/python -m unittest \
  test_vald test_get_moog_species test_line_profile_regressions \
  test_grid_regressions test_moog_imports test_moog_summary -v
```

The integration tests require external MOOG/Turbospectrum executables and
atmosphere grids configured in `~/.arcanesrc`. These are now configured and tested:

```sh
PYTHONPATH=test MPLCONFIGDIR=/tmp/arcane-mpl .venv/bin/python -m unittest \
  test_synthesis_atmospheres test_modelatm test_moog test_ts -v
```

Result: all 35 tests in the command above pass. `git diff --check` also passes.
