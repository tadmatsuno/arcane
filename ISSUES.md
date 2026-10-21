# Repository bug audit

## Fixed: Turbospectrum ignores the requested wavelength step

- `synth(dwvl_step=...)` omitted the argument when invoking opacity generation
  and synthesis, so all spectra used the default 0.01 Å spacing.
- Forward the requested step to both stages, including cached-opacity runs.
- Validation: `test/test_ts_sampling.py` reproduces the incorrect sample count
  and verifies real synthesis at 0.02 and 0.04 Å steps with new/cached opacity.

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

## Fixed: AVG3D radiation pressure is lost when reading saved atmospheres

- The final column name retained its trailing newline, so radiation pressure
  was stored under `Prad\n` rather than `Prad`. Strip column whitespace.
- Validation: `test/test_atmosphere_grids.py` checks interpolation, finite and
  physically positive structure fields, full profile round-trips for a solar
  atmosphere and two metal-poor atmospheres, and out-of-range rejection for
  both grids. All three AVG3D round-trips reproduced the missing field.

## Fixed: fitting tests override the configured MOOG executable

- Importing `test_fitting.py` attempted to replace the configured executable with
  an obsolete `/mnt/d/...` path, preventing the fitting tests from running and
  potentially rewriting user configuration. Use the existing configured path.

## Fixed: synthesis working directories and MOOG log isolation

- Both synthesis wrappers failed when a requested working directory did not
  already exist, including temporary-directory runs. Create it before writing
  input files or creating a temporary child directory.
- MOOG wrote every run's log in the caller's directory, causing parallel grid
  workers to overwrite one another's logs. Write it inside each run's working
  directory, including archived logs.
- Both programs now receive their working directory through `subprocess.run`;
  the Python process no longer temporarily changes its global working directory.
- Validation: `test/test_synthesis_workdirs.py` runs both real programs with
  new nested directories, in serial and temporary-directory modes, and verifies
  output, log placement, current directory, and temporary-directory cleanup.

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

## Final integration validation

With the user's executables and both atmosphere grids configured, the full
discovered suite passes: **72 tests**, including synthesis, abundance response,
fitting, parallel spectral grids, atmosphere interpolation, and serialization.

```sh
MPLCONFIGDIR=/tmp/arcane-mpl .venv/bin/python -m unittest discover -s test -p 'test*.py' -v
```

The captured result is `/tmp/arcane-integration-final.log`. Additional real
synthesis checks at Teff=5000 K, logg=2.5, [Fe/H]=-2, vt=1.5 km/s pass for all
four program/grid combinations in the 5853–5854 Å window. Each returns 101
ordered wavelengths and finite flux with an absorption line. The results are in
`/tmp/arcane-giant-synthesis.log`.

The suite still emits existing numerical warnings from logarithms of zero
pressure during interpolation and division by zero in a continuum-fitting
trial. The checked atmospheric profiles and final synthetic spectra are finite.
