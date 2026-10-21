"""Integration tests using configured synthesis programs and atmosphere grids."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcane.mdlatm import avg3d, marcs
from arcane.synthesis import moog, turbospectrum

DATA = Path(__file__).resolve().parent / 'DATA'


class TestSynthesisAtmospheres(unittest.TestCase):
    def test_both_programs_with_both_atmosphere_grids(self):
        for backend, filename in ((moog, 'lines_synth.in'),
                                  (turbospectrum, 'lines_ts.in')):
            for atmosphere in ('marcs', 'avg3d'):
                with self.subTest(backend=backend.__name__, atmosphere=atmosphere):
                    with tempfile.TemporaryDirectory() as workdir:
                        kwargs = dict(
                            linelist=str(DATA / filename), workdir=workdir,
                            teff=5777., logg=4.44, feh=0., vt=1.,
                            mdlatm_io=atmosphere, wmin=5853., wmax=5854.,
                            addtime_stamp=False,
                        )
                        wavelength, flux = backend.synth(**kwargs)
                        self.assertGreater(len(wavelength), 50)
                        self.assertTrue(np.all(np.diff(wavelength) > 0))
                        self.assertTrue(np.all(np.isfinite(flux)))
                        self.assertGreater(float(flux.min()), 0.)
                        self.assertLess(float(flux.min()), 0.6)
                        # The requested abundance must affect a real line.
                        _, enhanced = backend.synth(**kwargs, A_56=2.5)
                        index = np.argmin(np.abs(wavelength - 5853.67))
                        self.assertLess(enhanced[index], flux[index] - 0.02)

    def test_stagger_export_preserves_structure_and_depth_scale(self):
        # Alpha_R = ln(10) * tau_R / h gives dz/d(log tau_R) = h.
        logtau = np.linspace(-4., 2., 10)
        atmosphere = dict(
            lgTauR=logtau, AlphaTauR=np.log(10.) * 10.**logtau / 1e7,
            T=np.linspace(4000., 7000., 10), Density=np.full(10, 1e-8),
            ndepth=10, logg=4.44, teff=5777., m_h=0.,
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'stagger.mod'
            avg3d.write_stagger(filename, atmosphere)
            structure = np.loadtxt(filename, skiprows=2)
        np.testing.assert_allclose(structure[:, 0], (logtau - logtau[0]) * 1e7, atol=1e-6)
        np.testing.assert_allclose(structure[:, 1], atmosphere['T'])
        np.testing.assert_allclose(structure[:, 2], atmosphere['Density'])

    def test_turbospectrum_accepts_saved_avg3d_atmosphere(self):
        atmosphere = avg3d.get_model(5777., 4.44, 0.)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'avg3d.mod'
            atmosphere.write(filename)
            wavelength, flux = turbospectrum.synth(
                linelist=str(DATA / 'lines_ts.in'), workdir=directory,
                marcs_mod_file=str(filename), mdlatm_io='avg3d',
                feh=0., vt=1., wmin=5853., wmax=5854., addtime_stamp=False,
            )
            self.assertEqual(len(wavelength), len(flux))
            self.assertTrue(np.all(np.isfinite(flux)))
            self.assertLess(float(flux.min()), 0.6)


if __name__ == '__main__':
    unittest.main()
