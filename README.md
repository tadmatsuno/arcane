# arcane
Analyzing high-Resolution spectra for Chemical AbundaNcE

Tools that have been used by T. Matsuno for abundance analysis using high-resolution stellar spectra

Currently supported features are 
  - Continuum placement (continuum)
  - Equivalent widths measurements (ew)
  - Reading, writing, and interpolating model atmospheres (mdlatm) currently only for MARCS
  - More will be added...

## Repeated spectral calculations

`LineSynth` automatically reuses smoothing and rebinning geometry when its
wavelength grids stay unchanged. It refreshes the geometry when grids or radial
velocity change. The smoothing kernel still extends to 10 Gaussian sigma.

For repeated standalone calculations on fixed grids, prepare the geometry once:

```python
from arcane.utils import utils

smooth = utils.SmoothingPlan(synthetic_wavelength)  # FWHM in km/s
rebin = utils.RebinPlan(synthetic_wavelength, observed_wavelength,
                       conserve_count=False)  # flux density, rather than counts

observed_flux = rebin(smooth(synthetic_flux, vfwhm),
                     wavelength_scale=1.0 + rv / utils.ckm)
```

Flux and FWHM can change on every call. `RebinPlan` copies both grids; construct
a new plan if their values change. Changing `wavelength_scale` refreshes its
interpolation mapping. Nonfinite flux uses the general rebinning path because
excluding samples changes pixel boundaries. `SmoothingPlan(..., fwhm_vel=False)`
accepts FWHM in wavelength units. The existing `rebin` and `smooth_spectrum`
functions retain their interfaces.

Run the numerical comparison and performance benchmark against the original code:

```sh
MPLCONFIGDIR=/tmp/arcane-mpl .venv/bin/python scripts/benchmark_spectrum.py \
  --baseline-ref 390d515 --output /tmp/spectrum-performance.json
```

Timings isolate spectral processing from external synthesis programs and exclude
plan construction. They include repeated evaluations with changing radial velocity.
