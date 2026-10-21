"""Integration checks for isolated synthesis working directories."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcane.synthesis import moog, readvald, turbospectrum

DATA = Path(__file__).resolve().parent / 'DATA'


class TestSynthesisWorkdirs(unittest.TestCase):
    def test_new_working_directories(self):
        linelist = readvald.readvald(str(DATA / 'vald/Vald_stellar_short_hfs'))
        for backend in (moog, turbospectrum):
            for parallel in (False, True):
                with self.subTest(backend=backend.__name__, parallel=parallel):
                    with tempfile.TemporaryDirectory() as directory:
                        workdir = Path(directory) / 'new' / 'synthesis'
                        cwd = Path.cwd()
                        wavelength, flux = backend.synth(
                            linelist=linelist, workdir=str(workdir),
                            teff=5777., logg=4.44, feh=0., vt=1.,
                            in_parallel=parallel, addtime_stamp=False,
                        )
                        self.assertEqual(Path.cwd(), cwd)
                        self.assertEqual(len(wavelength), len(flux))
                        self.assertTrue(np.all(np.isfinite(flux)))
                        if parallel:
                            self.assertEqual(list(workdir.iterdir()), [])
                        elif backend is moog:
                            self.assertGreater((workdir / 'moog.log').stat().st_size, 0)
                        else:
                            self.assertTrue(list(workdir.glob('ts_*.log')))


if __name__ == '__main__':
    unittest.main()
