# Repository bug audit

## Installation: equivalent-width GUI imports

- The equivalent-width GUI imported the obsolete, undeclared `pycont` package
  and a top-level `iofiles` module. It now uses arcane's packaged utilities.
- An offscreen Qt regression test imports both GUIs and checks that the
  equivalent-width table and package-local file utilities are available.

## Installation validation

- `uv build` successfully builds a source archive and then a wheel from that
  archive. Build output is in `/tmp/arcane-installation-build.log`.
- Installed the final wheel with all dependencies into a fresh Python 3.12
  environment at `/tmp/arcane-install-check`. Running the full suite from
  `/tmp`, outside the source checkout, passes **107 tests**, including both GUIs,
  real MOOG/Turbospectrum synthesis, and both atmosphere grids. Output is in
  `/tmp/arcane-installed-wheel-suite.log`.
- Independently verified `uv pip install -e .` in a second fresh environment,
  with imports and spectral processing executed outside the checkout.
- `uv pip check` passes for both environments. `git diff --check` passes.
- Each issue was fixed on its own branch and merged into `codex-main`; the
  original `main` branch and the existing user edit to `.gitignore` are preserved.

## Installation: optional external configuration

- MARCS, AVG3D, and Turbospectrum imports required an existing configuration
  and grids, preventing use on a fresh installation. Imports now tolerate absent
  settings; grid and synthesis calls explain which setting is required.
- Path setters assumed the configuration already existed, so they could not
  perform initial setup. A shared configuration helper creates it on explicit
  setup, preserves other settings, and retains the existing backup convention.
- MARCS and AVG3D setters changed the path without reloading grid metadata.
  They now refresh the associated data together.
- Regression tests cover missing/partial configurations in fresh interpreters,
  first-time configuration, preserving settings/backups, and switching grids.

## Installation: distribution metadata and resources

- The source archive omitted `requirements.txt`, so the wheel built from it
  silently declared no dependencies. Dependencies now come from the shared
  requirements file through `pyproject.toml`, and that file ships in the archive.
- Building the package wrote `~/.arcanesrc`. The compatibility `setup.py` now
  delegates to setuptools without modifying user configuration.
- The wheel omitted the equivalent-width periodic table. Explicit package-data
  rules now include UI files, all scientific tables, and their references.
- Added uv/pip installation instructions and an external-data configuration
  template. Python dependencies and existing package APIs remain unchanged.
- Validation: `uv build` builds a wheel from the source archive; archive checks
  confirm all eight dependencies, runtime resources, and configuration template.

## Approved performance improvement: prepared rebinning geometry

- `RebinPlan` retains pixel edges, widths, chunk ranges, and interpolation
  brackets for repeated flux arrays on fixed grids. A Doppler-scale change
  rebuilds the mapping using the actual shifted wavelengths; nonfinite flux
  falls back to general rebinning because removing samples changes their edges.
- `LineSynth` reuses local plans for at most two target grids, supporting raw
  and binned continuum fits without an unbounded cache. Source-grid changes,
  including in-place mutation, discard the plans. Target grids are copied and
  compared before reuse so changes do not silently reuse stale geometry.
- Validation: independent pixel-overlap integrals, float32/float64 flux,
  repeated changing flux, count/flux modes, gaps, NaNs, chunking, changing
  velocities, source/target mutation, and existing fitting regressions.

## Approved performance improvement: reduce rebinning scans

- Compute the finite-flux mask once and avoid copying already finite inputs.
  Reuse source differences and skip gap-threshold medians when gap masking is
  disabled. Find output ranges with binary search and write contiguous slices
  rather than constructing and scanning a full output mask for every chunk.
- Chunks with no output overlap are skipped; fully nonoverlapping output is
  zero rather than raising an indexing error.
- Validation: exact agreement with the pre-optimization implementation across
  16 combinations of count/flux modes, NaNs, finite gaps, and chunk sizes.
  Independent pixel-overlap integration and gap/coverage tests also pass.

## Approved performance improvement: reuse smoothing geometry

- Removed unused Python min/max scans and calculations for the inactive
  smoothing mode. `SmoothingPlan` retains grid spacing and uniformity checks
  while accepting a different flux array and FWHM on every call.
- `LineSynth` reuses a local plan and refreshes it when source wavelengths
  change, including in-place mutation. The Gaussian filter, constant padding,
  and 10-sigma truncation remain unchanged.
- Validation: prepared and standalone smoothing match the original filter
  exactly in both modes, including zero width; warning and changing-grid tests
  pass along with the existing analytic and fitting regressions.

## Fixed: uncertainty estimation lacks fitting data and alters the original fit

- `fit` did not retain the observations required by `uncertainty`, which raised
  `AttributeError`. Supplying those attributes manually exposed another bug:
  uncertainty refits changed the original model, and rebinding local `self`
  could not restore it.
- Retain independent copies of the raw observations, continuum, and optimizer
  settings after fitting. Refit an independent model copy for each uncertainty
  sample, keeping the original state intact even when a callback fails.
- Raw sampling prevents repeated binning. Missing data and zero independent-
  noise samples now produce descriptive errors. Manual wavelength/flux inputs
  and the six deterministic continuum offsets remain supported.
- Validation: `test/test_uncertainty.py` checks deterministic continuum offsets,
  independent pixel noise, original-state preservation, copied raw data and
  binned refits, continuum reuse, callback failure, and invalid requests.

## Fixed: wavelength-based smoothing ignores its requested FWHM

- With `fwhm_vel=False`, smoothing used wavelength spacing itself as a Gaussian
  width in pixels, ignoring `vfwhm`. Different requested widths gave identical
  spectra, often with no broadening at all.
- Convert the supplied wavelength FWHM to Gaussian sigma and divide by pixel
  spacing. Velocity-based smoothing retains its existing behavior.
- Validation: `test/test_spectrum_smoothing.py` checks two widths against the
  analytic Gaussian convolution variance, equivalent-width conservation, zero
  width, and agreement between equivalent wavelength and velocity widths.

## Fixed: multi-file synthesis uses only the first line list's wavelength limits

- Turbospectrum filled missing limits inside the first iteration, skipping
  wavelength inference for subsequent line lists. This silently excluded lines
  outside the first file's range and made the result depend on file order.
- Infer missing limits from the union of every supplied file. Explicit lower
  and upper bounds retain precedence independently.
- Validation: `test/test_ts_multiple_linelist_windows.py` reproduces truncation
  in both file orders and with either bound supplied, then checks real synthesis
  over the correct union and preservation of fully explicit limits.

## Fixed: masked samples change synthetic-line bin widths during fitting

- Curve fitting evaluated synthetic spectra only at the retained wavelengths.
  Rebinning interprets those wavelengths as pixel centers, so sample gaps and
  sigma clipping enlarged adjacent bins. Optimization then fit a different
  model from the one evaluated on the full spectrum. Tiny noise could trigger
  clipping and turn an initially accurate fit into a severely incorrect result.
- Evaluate model spectra on the complete binned observation grid and then apply
  the fitting mask, with or without a continuum model.
- Validation: `test/test_fitting_sampling.py` reproduces failures with three
  fixed noise seeds and deterministic sample gaps, and verifies recovered line
  height, velocity, and width. The original unseeded integration test exposed
  this failure during the first full-suite run of this review.

## Fixed: continuum fitting crashes when observations are binned

- Deterministic-update and residual-based fits evaluated the line model on
  binned wavelengths, then divided the original unbinned flux by that shorter
  array while estimating the continuum. With `naverage > 1` this raised a shape
  mismatch instead of fitting.
- Evaluate the line model on the raw wavelengths for continuum estimation,
  matching the existing curve-fit path. The line fit still uses binned samples.
- Validation: `test/test_binned_continuum_fitting.py` reproduces both failing
  paths and verifies recovered depth, continuum, and finite fitted flux.

## Fixed: atmosphere interpolation mutates inputs and mishandles depth overlap

- Resampling changed the second input atmosphere in place and left the first
  untrimmed, causing shape errors for partially overlapping depth ranges.
- Resample both into new models on the common optical-depth grid. Reject
  insufficient overlap explicitly and leave both source atmospheres intact.
- Express logarithmic interpolation as the equivalent geometric product,
  preserving zero pressures and exact endpoints without `log(0)` or `0 * inf`.
- Validation: `test/test_atmosphere_interpolation.py` reproduces the shape error,
  source mutation, and zero-pressure endpoint errors. Real-grid interpolation
  and program integration are checked by the existing atmosphere/synthesis tests.

## Fixed: isotope inputs are rejected or silently ignored

- Both programs compared dictionary keys to the `str` class by identity,
  rejecting valid string keys. Validate with `isinstance` instead.
- Turbospectrum added `I_...` inputs only when overwriting an existing dictionary
  entry. Always add the requested isotope; retain the documented keyword-over-
  dictionary precedence and its warning.
- Validation: `test/test_isotope_inputs.py` reproduces both failures and checks
  the generated control files and finite spectra from the real executables.

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

## Initial integration validation

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

The later atmosphere-interpolation fix removes warnings from logarithms of
zero pressure. A continuum-fitting trial still emits an existing division-by-
zero warning; the checked atmospheric profiles and final spectra are finite.

## Further review validation

After fixes for requested wavelength spacing, isotope controls, atmosphere
interpolation, binned-continuum fitting, and masked-spectrum fitting, the full
discovered suite passes **82 tests** with the same command above. Output is
captured in `/tmp/arcane-review-verified.log`. `git diff --check` also passes.

The masked-spectrum fix was prompted by an intermittent failure in the first
full run. Its regression tests use three fixed noise seeds that reproduced
the failure and explicit sample gaps, avoiding reliance on a favorable random
test run. All five issues were fixed on separate branches and merged into
`codex-main`; the original `main` branch remains unchanged.

## Multi-file, smoothing, and uncertainty validation

The three subsequently approved fixes were implemented on separate issue
branches and merged into `codex-main`. The full discovered suite now passes
**93 tests**, including real Turbospectrum synthesis with multiple line lists,
analytic Gaussian smoothing checks, and uncertainty-refitting regressions.
The command remains:

```sh
MPLCONFIGDIR=/tmp/arcane-mpl .venv/bin/python -m unittest discover -s test -p 'test*.py' -v
```

Captured output: `/tmp/arcane-three-fixes-suite.log`. `git diff --check` passes.
No performance improvements were implemented as part of these bug fixes.

## Approved performance validation

The separately approved optimizations preserve the 10-sigma kernel and existing
public function signatures. The full discovered suite passes **103 tests**;
output is in `/tmp/arcane-performance-suite.log`. `git diff --check` passes.

The reproducible benchmark in `scripts/benchmark_spectrum.py` compares against
pre-optimization commit `390d515`. All **64 differential comparisons** agree
exactly (maximum absolute difference 0), including changes to flux, Doppler
scale, count/flux modes, gaps, NaNs, and chunking.

Representative repeated `LineSynth` evaluations on synthetic test spectra:

| Pixels | FWHM (km/s) | Previous (ms) | Optimized (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 5 | 0.232 | 0.074 | 3.1x |
| 100,000 | 5 | 11.988 | 2.088 | 5.7x |
| 100,000 | 50 | 30.625 | 21.285 | 1.4x |

For the 100,000-pixel, 5 km/s case, repeated prepared rebinning takes
0.498 ms versus 3.089 ms previously (6.2x); prepared smoothing takes 1.467 ms
versus 8.516 ms (5.8x). With changing velocity, combined evaluation takes
3.174 ms versus 11.971 ms (3.8x). Broad kernels still spend most time convolving.

These are local microbenchmarks, exclude initial plan construction, and use a
cheap analytic spectrum generator. They do not claim equivalent speedups for
external MOOG/Turbospectrum execution. Full timings and environment versions are
in `/tmp/arcane-spectrum-performance-results.json`; README documents usage and
the benchmark command.
