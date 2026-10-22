# arcane
Analyzing high-Resolution spectra for Chemical AbundaNcE

Tools that have been used by T. Matsuno for abundance analysis using high-resolution stellar spectra

Currently supported features are 
  - Continuum placement (continuum)
  - Equivalent widths measurements (ew)
  - Reading, writing, and interpolating MARCS and averaged 3D model atmospheres (mdlatm)
  - More will be added...

## Installation

From a checkout of this repository, use [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12
uv pip install .
```

This installs arcane and its Python dependencies, including PyQt5 for the GUI
tools, into `.venv`. Run Python with `.venv/bin/python`, or activate the
environment with `source .venv/bin/activate`. For development, replace the last
command with `uv pip install -e .` so source edits take effect immediately.

Standard pip installation is also supported:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Python 3.8 or newer is declared supported; Python 3.12 is the tested setup.
GUI tools need a graphical desktop and the system libraries required by Qt.

### Synthesis programs and atmosphere grids

MOOGSILENT, Turbospectrum, and the large atmosphere grids are installed
separately. Building or installing arcane does not modify your home directory.
For synthesis and grid interpolation, copy the configuration template:

```sh
cp .arcanesrc.example ~/.arcanesrc
```

Edit the paths in `~/.arcanesrc` to match your installations. Keep only the keys
for the features you use. `moog_silent` points to the compiled executable;
`turbospectrum_root` contains `DATA` and an `exec-ifx`, `exec-intel`, or `exec-gf`
directory with `babsma_lu` and `bsyn_lu`. The MARCS grid directory must contain
`MARCS_avai.dat`, `MARCS_grid.csv`, and the referenced models; the AVG3D directory
must contain `avg3d.sav`.

You can also configure paths from Python with `set_moogsilent_path`,
`set_turbospectrum_path`, `set_marcs_path`, and `set_avg3d_path` in their
respective `arcane.synthesis` and `arcane.mdlatm` modules. These preserve other
settings and back up an existing configuration to `~/.arcanesrc_old`.

To build distributable source and wheel archives, run `uv build`. A wheel can
be installed with `uv pip install /path/to/arcane-0.0.1-py3-none-any.whl`.

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
