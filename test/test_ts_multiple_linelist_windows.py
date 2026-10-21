"""All supplied line lists must contribute to inferred synthesis limits."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcane.synthesis import turbospectrum

DATA = Path(__file__).resolve().parent / 'DATA'


class TestMultipleLinelistWindows(unittest.TestCase):
    def test_union_limits_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            files = []
            for label, wavelength in (('blue', 5000.), ('red', 5010.)):
                filename = directory / f'{label}.lin'
                filename.write_text("'26.0000' 1 1\nFe I\n" +
                                    f"{wavelength:.3f} 2.0 -1.0 3.0 1.0 1.0e6 'x' 'x' 0.0 0.0\n")
                files.append(str(filename))
            cases = [
                (files, {}, (4998., 5012.)),
                (files[::-1], {}, (4998., 5012.)),
                (files, {'wmin': 5001.}, (5001., 5012.)),
                (files[::-1], {'wmax': 5009.}, (4998., 5009.)),
                (files, {'wmin': 5004., 'wmax': 5006.}, (5004., 5006.)),
            ]
            for index, (linelists, bounds, expected) in enumerate(cases):
                with self.subTest(order=linelists, bounds=bounds):
                    wavelength, flux = turbospectrum.synth(
                        linelist=linelists, workdir=str(directory / f'run{index}'),
                        marcs_mod_file=str(DATA / 'sun_marcs.mod'),
                        vt=1., feh=0., addtime_stamp=False, dwvl_step=0.1,
                        **bounds,
                    )
                    np.testing.assert_allclose(wavelength[[0, -1]], expected, atol=1e-8)
                    self.assertTrue(np.all(np.isfinite(flux)))


if __name__ == '__main__':
    unittest.main()
