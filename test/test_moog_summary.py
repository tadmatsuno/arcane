"""MOOG output parsing without a MOOG executable."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcane.synthesis import moog


class TestMoogSummary(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'moog.sum'

    def test_ewfind_reads_all_rows(self):
        rows = [[5000., 2., -1., 26., 7.5, 20.],
                [5001., 3., -2., 26., 7.5, 10.]]
        self.path.write_text('wavelength\n' + ''.join(
            ''.join(f'{value:10.3f}' for value in row) + '\n' for row in rows
        ))
        result = moog.read_moog_sum_ewfind(self.path)
        np.testing.assert_allclose(result['wavelength'], [5000., 5001.])
        np.testing.assert_allclose(result['ew'], [20., 10.])

    def test_abfind_reads_all_rows_and_species_blocks(self):
        def row(wavelength, species, abundance):
            return ''.join(f'{value:10.3f}' for value in
                           [wavelength, species, 2., -1., 20., -5., abundance, 0.]) + '\n'
        self.path.write_text(
            'wavelength\n' + row(5000., 26., 7.4) + row(5001., 26., 7.6) +
            'average abundance = 7.5\n\n' +
            'Abundance Results for Species Ti I\n' +
            'wavelength\n' + row(5002., 22., 5.) + 'average abundance = 5.0\n'
        )
        result = moog.read_moog_sum_abfind(self.path)
        np.testing.assert_allclose(result['wavelength'], [5000., 5001., 5002.])
        np.testing.assert_allclose(result['abund'], [7.4, 7.6, 5.])

    def test_synth_reads_valid_summary(self):
        self.path.write_text('header\nMODEL atmosphere\n5000 5000.02 0.01 1\n'
                             ' 0.1000 0.2000 0.3000\n')
        wavelength, flux = moog.read_moog_sum_synth(self.path)
        np.testing.assert_allclose(wavelength, [5000., 5000.01, 5000.02])
        np.testing.assert_allclose(flux, [0.1, 0.2, 0.3])

    def test_synth_rejects_missing_model_header(self):
        for contents in ('', 'MOOG failed before writing a model\n'):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                with self.assertRaisesRegex(ValueError, 'MODEL header'):
                    moog.read_moog_sum_synth(self.path)


if __name__ == '__main__':
    unittest.main()
