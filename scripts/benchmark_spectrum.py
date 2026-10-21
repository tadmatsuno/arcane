"""Benchmark numerical agreement and repeated spectral operations against git.

Run with .venv/bin/python scripts/benchmark_spectrum.py --output /tmp/spectrum.json
Timings exclude plan construction and isolate the numerical processing from
external synthesis programs. --baseline-ref identifies the pre-optimization code.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import timeit
import types

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arcane.spectrum.model import LineSynth
from arcane.utils import utils


def baseline_module(ref, filename, name):
    source = subprocess.check_output(['git', 'show', f'{ref}:{filename}'], cwd=ROOT, text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module  # dataclasses resolve annotations through the module.
    exec(compile(source, f'{ref}:{filename}', 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='390d515')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 100000])
    parser.add_argument('--widths', type=float, nargs='+', default=[5., 50.])
    parser.add_argument('--number', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    previous = baseline_module(args.baseline_ref, 'arcane/utils/utils.py', '_benchmark_old_utils')
    previous_model = baseline_module(args.baseline_ref, 'arcane/spectrum/model.py', '_benchmark_old_model')
    previous_model.utils = previous

    def milliseconds(function):
        function()  # warm up allocations and prepared geometry
        return min(timeit.repeat(function, number=args.number, repeat=args.repeat))*1000./args.number

    # Compare the generic and prepared paths against the actual previous code.
    rng = np.random.RandomState(43)
    max_difference = 0.
    comparisons = 0
    for counts in (False, True):
        for gap in (np.inf, 3.):
            for chunk in (1000000, 97):
                source = np.cumsum(rng.uniform(.05, .15, 1000))+5000.
                source[500:] += 10.
                target = np.linspace(source[5], source[-6], 500)
                kwargs = dict(conserve_count=counts, npix_mask_criterion=gap, nchunk_max=chunk)
                plan = utils.RebinPlan(source, target, **kwargs)
                for scale, missing in ((1., False), (1.00001, False), (.99999, True), (1., False)):
                    flux = rng.uniform(.1, 2., len(source))
                    if missing:
                        flux[[123, 345, 723]] = np.nan
                    expected = previous.rebin(source*scale, flux, target, **kwargs)
                    for actual in (utils.rebin(source*scale, flux, target, **kwargs),
                                   plan(flux, wavelength_scale=scale)):
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                        max_difference = max(max_difference, float(np.max(np.abs(actual-expected))))
                        comparisons += 1

    results = []
    for size in args.sizes:
        wavelength = 5000.*np.exp(np.arange(size)*.6/utils.ckm)
        profile = np.exp(-.5*((np.arange(size)-size//2)/10.)**2)
        flux = 1.-.7*profile
        target = wavelength[5:-5:2]
        rebin_plan = utils.RebinPlan(wavelength, target, conserve_count=False)
        smoothing_plan = utils.SmoothingPlan(wavelength)
        def synth(depth):
            return wavelength, 1.-depth*profile
        # Rebinning does not depend on the smoothing width; time it once per grid.
        rebin_times = {
            'rebin_baseline_ms': milliseconds(lambda: previous.rebin(wavelength, flux, target, conserve_count=False)),
            'rebin_ms': milliseconds(lambda: utils.rebin(wavelength, flux, target, conserve_count=False)),
            'rebin_prepared_ms': milliseconds(lambda: rebin_plan(flux)),
        }
        for width in args.widths:
            old_model = previous_model.LineSynth(synth, {'depth': .7}, ['depth'], vfwhm_in=width)
            new_model = LineSynth(synth, {'depth': .7}, ['depth'], vfwhm_in=width)
            for rv in (0., .3, -.3, 0.):
                old_model.model_parameters['rv'] = rv
                new_model.model_parameters['rv'] = rv
                np.testing.assert_allclose(new_model(target), old_model(target), rtol=1e-12, atol=1e-12)
            expected = previous.smooth_spectrum(wavelength, flux, width)
            np.testing.assert_array_equal(smoothing_plan(flux, width), expected)
            row = dict(pixels=size, fwhm_km_s=width)
            row.update(rebin_times)
            calls = {
                'smoothing_baseline_ms': lambda: previous.smooth_spectrum(wavelength, flux, width),
                'smoothing_ms': lambda: utils.smooth_spectrum(wavelength, flux, width),
                'smoothing_prepared_ms': lambda: smoothing_plan(flux, width),
                'line_evaluate_baseline_ms': lambda: old_model(target),
                'line_evaluate_ms': lambda: new_model(target),
            }
            for label, function in calls.items():
                row[label] = milliseconds(function)
            for label, model in (('moving_rv_baseline_ms', old_model), ('moving_rv_ms', new_model)):
                counter = [0]
                def moving():
                    counter[0] += 1
                    model.model_parameters['rv'] = (counter[0] % 3-1)*.3
                    return model(target)
                row[label] = milliseconds(moving)
            results.append(row)
    report = dict(baseline_ref=args.baseline_ref, numpy=np.__version__, scipy=scipy.__version__,
                  numerical_comparisons=comparisons, max_rebin_difference=max_difference,
                  results=results)
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output+'\n')
    print(output)


if __name__ == '__main__':
    main()
