"""Prepared mappings must follow flux validity, Doppler shifts, and grid changes."""
import unittest

import numpy as np

from arcane.spectrum.model import LineSynth
from arcane.utils import utils
from test_rebinning import overlap_integrals


class TestPreparedRebinning(unittest.TestCase):
    def test_irregular_grids_match_pixel_overlap_integrals(self):
        rng = np.random.RandomState(53)
        for dtype in (np.float64, np.float32):
            source = np.cumsum(rng.uniform(0.05, 0.15, 100)).astype(dtype)
            target = np.linspace(source[5], source[-6], 47).astype(dtype)
            for counts in (False, True):
                plan = utils.RebinPlan(source, target, conserve_count=counts)
                for _ in range(3):
                    flux = rng.uniform(0.1, 2., len(source)).astype(dtype)
                    with self.subTest(dtype=dtype, counts=counts):
                        expected = utils.rebin(source, flux, target, conserve_count=counts)
                        np.testing.assert_allclose(plan(flux), expected, rtol=1e-12, atol=1e-12)
                        if dtype is np.float64:
                            integrated = overlap_integrals(source, flux, target, counts)
                            np.testing.assert_allclose(plan(flux), integrated, rtol=1e-12, atol=1e-12)

    def test_shifted_chunked_gapped_and_nonfinite_flux(self):
        rng = np.random.RandomState(61)
        source = np.cumsum(rng.uniform(0.05, 0.15, 1000))+5000.
        source[500:] += 10.
        target = np.linspace(source[5], source[-6], 500)
        for counts in (True, False):
            for gap in (np.inf, 3.):
                for chunk in (1000000, 97):
                    kwargs = dict(conserve_count=counts, npix_mask_criterion=gap, nchunk_max=chunk)
                    plan = utils.RebinPlan(source, target, **kwargs)
                    for scale, missing in ((1., False), (1.00001, False), (0.99999, True), (1., False)):
                        flux = rng.uniform(0.1, 2., len(source))
                        if missing:
                            flux[[123, 345, 723]] = np.nan
                        with self.subTest(counts=counts, gap=gap, chunk=chunk, scale=scale, missing=missing):
                            expected = utils.rebin(source*scale, flux, target, **kwargs)
                            np.testing.assert_allclose(plan(flux, wavelength_scale=scale), expected,
                                                       rtol=1e-12, atol=1e-12)

    def test_plan_copies_grids_and_returns_zero_outside_coverage(self):
        source, target = np.arange(20.), np.arange(30., 40.)
        plan = utils.RebinPlan(source, target)
        source[:] = -1.
        target[:] = -1.
        np.testing.assert_array_equal(plan(np.ones(20)), np.zeros(10))

    def test_synthesis_refreshes_source_target_and_velocity_geometry(self):
        source = np.linspace(4998., 5002., 1001)
        target = np.linspace(4999., 5001., 101)
        def synth(depth):
            return source, 1.-depth*np.exp(-0.5*((source-5000.)/0.1)**2)
        fitted = LineSynth(synth, {'depth': 0.4}, ['depth'], vfwhm_in=5.)
        for step in range(5):
            if step == 1:
                target[:] += 0.01
            if step == 2:
                source[:] = np.linspace(4997., 5003., 1001)
            fitted.model_parameters['rv'] = step*0.3
            fitted.model_parameters['depth'] = 0.4+step*0.02
            actual = fitted(target)
            w, flux = synth(fitted.model_parameters['depth'])
            expected = utils.rebin(w*(1.+step*0.3/utils.ckm), utils.smooth_spectrum(w, flux, 5.),
                                   target, conserve_count=False)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        # Different output grids remain isolated and do not grow an unbounded cache.
        for count in (50, 75, 100, 125):
            fitted(np.linspace(4999., 5001., count))
        self.assertLessEqual(len(fitted._rebin_plans), 2)


if __name__ == '__main__':
    unittest.main()
