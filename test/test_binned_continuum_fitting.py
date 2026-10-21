"""Binned fits must estimate the continuum on compatible wavelength arrays."""
import unittest

import numpy as np

from arcane.spectrum.model import ModelBase, LineProfile, ContinuumPolynomial


class NormalizedLineProfile(LineProfile):
    def evaluate(self, wavelength, parameters):
        return 1. - super().evaluate(wavelength, parameters)


class ResidualLine(ModelBase):
    def __init__(self):
        super().__init__(naverage=3, niterate=1)
        self.x0_default = [0.2]
        self.bounds_default = ([0.], [1.])
        self.model_parameters = {'depth': 0.2}

    def evaluate(self, wavelength, parameters):
        return 1. - parameters['depth'] * np.exp(-0.5 * ((wavelength - 5000.) / 0.1)**2)

    def func_residual(self, wavelength, flux, parameters):
        self.model_parameters['depth'] = parameters[0]
        return self(wavelength) - flux


class TestBinnedContinuumFitting(unittest.TestCase):
    def test_binned_update_and_residual_fits_with_raw_continuum(self):
        wavelength = np.linspace(4999., 5001., 600)
        continuum_flux = 1.1 + 0.02 * (wavelength - 5000.)
        truth = NormalizedLineProfile([5000.], initial_depth=0.4, initial_fwhm=0.235482,
                            voigt=False, fix_dwvl=True, fix_fwhm=True)
        flux = truth(wavelength) * continuum_flux
        models = [
            NormalizedLineProfile([5000.], initial_depth=0.2, initial_fwhm=0.235482,
                        voigt=False, fix_dwvl=True, fix_fwhm=True,
                        naverage=3, niterate=1),
            ResidualLine(),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                continuum = ContinuumPolynomial(
                    order=1, niterate=1, samples=[[4999., 4999.5], [5000.5, 5001.]],
                )
                result = model.fit(wavelength, flux, continuum_model=continuum)
                self.assertEqual(len(result.wavelength), 200)
                np.testing.assert_allclose(result.model_parameters['depth'], 0.4, atol=0.001)
                np.testing.assert_allclose(continuum(wavelength), continuum_flux, atol=1e-5)
                self.assertTrue(np.all(np.isfinite(result.flux_fit)))
                np.testing.assert_array_equal(result.wavelength_raw, wavelength)


if __name__ == '__main__':
    unittest.main()
