"""Smoothing widths are measured against Gaussian convolution identities."""
import unittest

import numpy as np

from arcane.utils import utils


class TestSpectrumSmoothing(unittest.TestCase):
    def setUp(self):
        self.wavelength = np.linspace(4997., 5003., 3001)
        self.offset = self.wavelength - 5000.
        self.flux = 1. - 0.5 * np.exp(-0.5 * (self.offset / 0.02)**2)

    def test_wavelength_fwhm_controls_broadening_and_preserves_area(self):
        area = np.sum(1. - self.flux)
        for fwhm in (0.1, 0.5):
            with self.subTest(fwhm=fwhm):
                smoothed = utils.smooth_spectrum(self.wavelength, self.flux, fwhm,
                                                 fwhm_vel=False)
                absorption = 1. - smoothed
                sigma = np.sqrt(np.sum(absorption * self.offset**2) / np.sum(absorption))
                expected = np.sqrt(0.02**2 + (fwhm / np.sqrt(8. * np.log(2.)))**2)
                np.testing.assert_allclose(sigma, expected, rtol=1e-6)
                np.testing.assert_allclose(np.sum(absorption), area, rtol=1e-10)

    def test_zero_width_preserves_the_spectrum(self):
        smoothed = utils.smooth_spectrum(self.wavelength, self.flux, 0., fwhm_vel=False)
        np.testing.assert_array_equal(smoothed, self.flux)

    def test_wavelength_and_velocity_widths_agree(self):
        velocity_flux = utils.smooth_spectrum(self.wavelength, self.flux, 10.)
        wavelength_flux = utils.smooth_spectrum(
            self.wavelength, self.flux, 5000. * 10. / utils.ckm, fwhm_vel=False,
        )
        np.testing.assert_allclose(wavelength_flux, velocity_flux, rtol=1e-9, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
