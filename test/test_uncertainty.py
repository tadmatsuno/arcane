"""Uncertainty refits must preserve the fitted model and its sampling."""
import unittest
from unittest.mock import patch

import numpy as np

from arcane.spectrum.model import ContinuumPolynomial, LineSynth
from arcane.spectrum import model as model_module


class RecordingPolynomial(ContinuumPolynomial):
    def update(self, wavelength, flux):
        self.last_sample_count = len(wavelength)
        return super().update(wavelength, flux)


def absorption_spectrum(depth):
    wavelength = np.arange(4998., 5002., 0.01)
    return wavelength, 1. - depth * np.exp(-0.5 * ((wavelength - 5000.) / 0.1)**2)


class TestUncertainty(unittest.TestCase):
    def make_model(self, naverage=1):
        wavelength = np.linspace(4999., 5001., 12)
        flux = 1. + 0.001 * np.sin(np.linspace(0., 2. * np.pi, 12))
        model = RecordingPolynomial(order=0, niterate=1, naverage=naverage)
        model.fit(wavelength, flux)
        return model, wavelength, flux

    def test_continuum_offsets_after_fit_preserve_original_state(self):
        model, wavelength, flux = self.make_model()
        polynomial = model.model_parameters['poly']
        result = model.uncertainty(lambda m: {'level': float(m(5000.))},
                                   std=0.1, nmc=0, iid=False)
        np.testing.assert_allclose(result['level'], [0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
        self.assertIs(model.model_parameters['poly'], polynomial)
        np.testing.assert_array_equal(model.wavelength, wavelength)
        np.testing.assert_array_equal(model.flux, flux)

    def test_independent_pixel_noise_uses_requested_samples(self):
        model, wavelength, flux = self.make_model()
        noise = [np.ones(len(flux)), -np.ones(len(flux))]
        with patch('arcane.spectrum.model.np.random.randn', side_effect=noise):
            result = model.uncertainty(lambda m: {'level': float(m(5000.))},
                                       std=0.1, nmc=2, iid=True)
        np.testing.assert_allclose(result['level'], [1.1, 0.9])
        np.testing.assert_allclose(model(5000.), 1.)

    def test_manually_supplied_observations_remain_supported(self):
        model = ContinuumPolynomial(order=0, niterate=1)
        model.wavelength = np.linspace(4999., 5001., 12)
        model.flux = 1. + 0.001 * np.sin(np.linspace(0., 2. * np.pi, 12))
        result = model.uncertainty(lambda m: {'level': float(m(5000.))},
                                   std=0., nmc=0, iid=False)
        np.testing.assert_allclose(result['level'], np.ones(6))
        self.assertIsNone(model.model_parameters['poly'])

    def test_binned_refits_start_from_copied_raw_data(self):
        model, wavelength, flux = self.make_model(naverage=3)
        original_wavelength, original_flux = wavelength.copy(), flux.copy()
        wavelength[:] = -1.
        flux[:] = -1.
        result = model.uncertainty(lambda m: {'samples': m.last_sample_count},
                                   std=0., nmc=0, iid=False)
        self.assertEqual(result['samples'], [4] * 6)
        np.testing.assert_array_equal(model.wavelength, original_wavelength)
        np.testing.assert_array_equal(model.flux, original_flux)

    def test_callback_failure_preserves_original_model(self):
        model, wavelength, flux = self.make_model()
        polynomial = model.model_parameters['poly']
        def fail(sampled_model):
            raise RuntimeError('callback failure')
        with self.assertRaisesRegex(RuntimeError, 'callback failure'):
            model.uncertainty(fail, std=0.1, nmc=0, iid=False)
        self.assertIs(model.model_parameters['poly'], polynomial)
        np.testing.assert_array_equal(model.flux, flux)

    def test_refits_reuse_continuum_and_optimizer_settings(self):
        wavelength = np.arange(4999., 5001., 0.02)
        truth = LineSynth(absorption_spectrum, {'depth': 0.4}, ['depth'],
                          vfwhm_in=0., rv_in=0.)
        model = LineSynth(absorption_spectrum, {'depth': 0.2}, ['depth'],
                          vfwhm_in=0., rv_in=0., niterate=1)
        continuum = ContinuumPolynomial(order=0, niterate=1,
                                        samples=[[4999., 4999.4], [5000.6, 5001.]])
        model.fit(wavelength, truth(wavelength) * 1.1, continuum_model=continuum,
                  optimization_parameter={'max_nfev': 250})
        polynomial = continuum.model_parameters['poly']
        with patch('arcane.spectrum.model.curve_fit', wraps=model_module.curve_fit) as optimizer:
            result = model.uncertainty(
                lambda m: {'depth': m.model_parameters['depth']}, std=0., nmc=1,
            )
        self.assertGreater(optimizer.call_count, 0)
        for call in optimizer.call_args_list:
            self.assertEqual(call.kwargs['max_nfev'], 250)
        np.testing.assert_allclose(result['depth'], [0.4], atol=1e-5)
        self.assertIs(continuum.model_parameters['poly'], polynomial)
        np.testing.assert_allclose(model.model_parameters['depth'], 0.4, atol=1e-5)

    def test_missing_data_and_zero_iid_samples_have_clear_errors(self):
        unfitted = ContinuumPolynomial(order=0)
        with self.assertRaisesRegex(ValueError, 'fit|wavelength'):
            unfitted.uncertainty(lambda m: {}, std=0.1)
        model, _, _ = self.make_model()
        with self.assertRaisesRegex(ValueError, 'nmc'):
            model.uncertainty(lambda m: {}, std=0.1, nmc=0, iid=True)


if __name__ == '__main__':
    unittest.main()
