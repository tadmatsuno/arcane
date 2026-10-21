"""Wavelength sampling is a user-visible synthesis setting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcane.synthesis import turbospectrum

DATA = Path(__file__).resolve().parent / 'DATA'


class TestTurbospectrumSampling(unittest.TestCase):
    def test_requested_step_is_used_with_new_and_cached_opacity(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                linelist=str(DATA / 'lines_ts.in'), workdir=directory,
                marcs_mod_file=str(DATA / 'sun_marcs.mod'),
                vt=1., feh=0., wmin=5853., wmax=5854.,
                run_id='sampling', addtime_stamp=False,
            )
            wavelength, flux = turbospectrum.synth(**kwargs, dwvl_step=0.02)
            self.assertEqual(len(wavelength), 51)
            np.testing.assert_allclose(np.diff(wavelength), 0.02, atol=1e-8)
            self.assertTrue(np.all(np.isfinite(flux)))
            opacity = next(Path(directory).glob('*.opac'))
            wavelength, flux = turbospectrum.synth(
                **kwargs, dwvl_step=0.04, ts_opac_file=str(opacity),
            )
            self.assertEqual(len(wavelength), 26)
            np.testing.assert_allclose(np.diff(wavelength), 0.04, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
