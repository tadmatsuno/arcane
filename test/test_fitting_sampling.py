"""Rejecting observations must not change the model's wavelength bins."""
import unittest

import numpy as np

from arcane.spectrum.model import ContinuumPolynomial, LineSynth
from arcane.utils import utils


def gaussian_spectrum(aheight):
    offset = np.arange(-2., 2., 0.01)
    return offset + 5000., aheight * np.exp(-0.5 * (offset / 0.1)**2)


def absorption_spectrum(aheight):
    wavelength, line = gaussian_spectrum(aheight)
    return wavelength, 1. - line


class TestFittingSampling(unittest.TestCase):
    def test_sigma_clipping_keeps_synthetic_line_parameters_stable(self):
        wavelength = np.arange(-1.5, 1.5, 0.02) + 5000.13
        synth_wavelength, flux = gaussian_spectrum(1.)
        flux = utils.smooth_spectrum(synth_wavelength, flux, 10.)
        flux = utils.rebin(synth_wavelength, flux, wavelength, conserve_count=False)
        for seed in (3, 17, 21):
            with self.subTest(seed=seed):
                observed = flux + 1e-6 * np.random.RandomState(seed).randn(len(flux))
                fitted = LineSynth(gaussian_spectrum, {'aheight': 0.5},
                                   ['aheight', 'rv', 'vFWHM'], vfwhm_in=5., rv_in=1.)
                result = fitted.fit(wavelength, observed)
                np.testing.assert_allclose(result.model_parameters['aheight'], 1., atol=0.001)
                np.testing.assert_allclose(result.model_parameters['rv'], 0., atol=0.001)
                np.testing.assert_allclose(result.model_parameters['vFWHM'], 10., atol=0.001)

    def test_sample_gaps_with_and_without_continuum(self):
        wavelength = np.arange(4999., 5001., 0.02)
        truth = LineSynth(absorption_spectrum, {'aheight': 0.4}, ['aheight'],
                          vfwhm_in=10., rv_in=0.)
        for with_continuum in (False, True):
            with self.subTest(continuum=with_continuum):
                continuum = ContinuumPolynomial(
                    order=0, niterate=1, samples=[[4999., 4999.3], [5000.7, 5001.]],
                ) if with_continuum else None
                flux = truth(wavelength) * (1.1 if with_continuum else 1.)
                fitted = LineSynth(
                    absorption_spectrum, {'aheight': 0.2}, ['aheight', 'vFWHM'],
                    vfwhm_in=5., rv_in=0., niterate=1,
                    samples=[[4999., 4999.96], [5000.04, 5001.]],
                )
                result = fitted.fit(wavelength, flux, continuum_model=continuum)
                np.testing.assert_allclose(result.model_parameters['aheight'], 0.4, atol=1e-4)
                np.testing.assert_allclose(result.model_parameters['vFWHM'], 10., atol=0.001)


if __name__ == '__main__':
    unittest.main()
