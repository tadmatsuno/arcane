"""Cached geometry must preserve smoothing and refresh for changing grids."""
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from scipy.ndimage import gaussian_filter

from arcane.spectrum.model import LineSynth
from arcane.utils import utils


class TestPreparedSmoothing(unittest.TestCase):
    def test_plan_matches_original_filter_and_tracks_flux_and_width(self):
        rng = np.random.RandomState(23)
        for velocity in (False, True):
            wavelength = np.linspace(4998., 5002., 1001)
            spacing = np.diff(wavelength)
            if velocity:
                spacing = spacing/(0.5*(wavelength[1:]+wavelength[:-1]))
            spacing = np.median(spacing) * (utils.ckm if velocity else 1.)
            plan = utils.SmoothingPlan(wavelength, fwhm_vel=velocity)
            for width in (0., 0.2, 5., 10.):
                flux = rng.uniform(0.1, 1., len(wavelength))
                sigma = (width/np.sqrt(8.*np.log(2.)))/spacing if velocity else width/(np.sqrt(8.*np.log(2.))*spacing)
                expected = gaussian_filter(flux, sigma, mode='constant', cval=1., truncate=10.)
                np.testing.assert_array_equal(plan(flux, width), expected)
                np.testing.assert_array_equal(utils.smooth_spectrum(wavelength, flux, width, velocity), expected)

    def test_nonuniform_grid_warning_is_preserved(self):
        wavelength = np.array([5000., 5000.01, 5000.03, 5000.1])
        for velocity in (False, True):
            output = StringIO()
            with redirect_stdout(output):
                plan = utils.SmoothingPlan(wavelength, fwhm_vel=velocity)
                plan(np.ones(4), 1.)
            self.assertIn('highly recommended to rebin', output.getvalue())

    def test_synthesis_refreshes_geometry_when_source_grid_is_mutated(self):
        wavelength = np.linspace(4998., 5002., 1001)
        def synth(depth):
            return wavelength, 1.-depth*np.exp(-0.5*((wavelength-5000.)/0.1)**2)
        fitted = LineSynth(synth, {'depth': 0.4}, ['depth'], vfwhm_in=5.)
        observed = np.linspace(4999., 5001., 101)
        fitted(observed)
        wavelength[:] = np.linspace(4997., 5003., 1001)
        actual = fitted(observed)
        w, flux = synth(0.4)
        expected = utils.rebin(w, utils.smooth_spectrum(w, flux, 5.), observed, conserve_count=False)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
