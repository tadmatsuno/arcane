"""Line-profile regressions that do not need external synthesis programs."""
import unittest

import numpy as np

from arcane.spectrum.model import LineProfile


class TestLineProfileRegressions(unittest.TestCase):
    def test_shared_width_and_shift(self):
        wavelength = np.linspace(4999.5, 5001.5, 501)
        truth = LineProfile([5000., 5001.], voigt=False)
        truth.model_parameters['depth'][:] = [0.3, 0.5]
        truth.model_parameters['fwhm'][:] = 0.16
        truth.model_parameters['dwvl'][:] = 0.035
        fitted = LineProfile(
            [5000., 5001.], voigt=False, share_fwhm=True,
            share_dwvl=True, niterate=1,
        )
        result = fitted.fit(wavelength, truth(wavelength))
        np.testing.assert_allclose(result.model_parameters['dwvl'], 0.035, atol=1e-4)
        np.testing.assert_allclose(result.model_parameters['fwhm'], 0.16, atol=1e-4)
        np.testing.assert_allclose(result.flux_fit, truth(wavelength), atol=1e-4)

    def test_shared_width_keeps_fixed_line(self):
        wavelength = np.linspace(4999.5, 5001.5, 501)
        truth = LineProfile([5000., 5001.], voigt=False)
        truth.model_parameters['fwhm'][:] = [0.1, 0.16]
        fitted = LineProfile(
            [5000., 5001.], voigt=False, share_fwhm=True,
            fix_fwhm=[True, False], fix_dwvl=True, niterate=1,
        )
        result = fitted.fit(wavelength, truth(wavelength))
        np.testing.assert_allclose(result.model_parameters['fwhm'], [0.1, 0.16], atol=1e-4)

    def test_add_and_delete_preserve_line_arrays_and_fitting(self):
        profile = LineProfile([5000., 5001.], voigt=False, fix_dwvl=True)
        profile.model_parameters['depth'][:] = [0.2, 0.4]
        profile.delete_line(0)
        np.testing.assert_allclose(profile.model_parameters['depth'], [0.4])
        profile.add_line2(5002.)
        np.testing.assert_allclose(profile.model_parameters['depth'], [0.4, 0.3])
        for key in ('center', 'dwvl', 'depth', 'fwhm', 'fgfwhm'):
            self.assertEqual(len(profile.model_parameters[key]), profile.nlines)
        for key in ('fix_dwvl', 'fix_fwhm', 'voigt'):
            self.assertEqual(len(profile.fit_control[key]), profile.nlines)
        wavelength = np.linspace(5000.5, 5002.5, 501)
        flux = profile(wavelength)
        result = profile.fit(wavelength, flux)
        np.testing.assert_allclose(result.flux_fit, flux, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
