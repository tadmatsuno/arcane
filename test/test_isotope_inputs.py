"""Check the isotope controls sent to the real synthesis programs."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from arcane.synthesis import moog, turbospectrum

DATA = Path(__file__).resolve().parent / 'DATA'


class TestIsotopeInputs(unittest.TestCase):
    def test_moog_accepts_documented_isotope_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            wavelength, flux = moog.synth(
                linelist=str(DATA / 'lines_synth.in'),
                moog_mod_file=str(DATA / 'model.in'), workdir=directory,
                addtime_stamp=False, Isotope_dict={'106.00113': 0.01},
            )
            control = (Path(directory) / 'batch.par').read_text()
            self.assertIn('106.00113 100.000', control)
            self.assertTrue(np.all(np.isfinite(flux)))

    def test_turbospectrum_accepts_dictionary_and_keyword_isotopes(self):
        for isotope_args, value in (({'Isotope_dict': {'6.012': 0.9}}, 0.9),
                                    ({'I_6_012': 0.9}, 0.9),
                                    ({'Isotope_dict': {'6.012': 0.8}, 'I_6_012': 0.9}, 0.9)):
            with self.subTest(arguments=isotope_args):
                with tempfile.TemporaryDirectory() as directory:
                    with warnings.catch_warnings(record=True) as emitted:
                        wavelength, flux = turbospectrum.synth(
                            linelist=str(DATA / 'lines_ts.in'),
                            marcs_mod_file=str(DATA / 'sun_marcs.mod'),
                            workdir=directory, run_id='isotopes',
                            addtime_stamp=False, vt=1., feh=0., **isotope_args,
                        )
                    for filename in ('isotopes.ts_input', 'isotopes_babsma.ts_input'):
                        control = (Path(directory) / filename).read_text()
                        self.assertIn("'ISOTOPES:' '1'", control)
                        self.assertIn(f'6.012 {value}\n', control)
                    self.assertTrue(np.all(np.isfinite(flux)))
                    if 'Isotope_dict' in isotope_args and 'I_6_012' in isotope_args:
                        self.assertTrue(any('prioritized' in str(item.message) for item in emitted))


if __name__ == '__main__':
    unittest.main()
