"""Check real atmosphere interpolation, serialization, and range validation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcane.mdlatm import avg3d, marcs


class TestAtmosphereGrids(unittest.TestCase):
    def test_interpolation_and_all_profile_roundtrips(self):
        for module in (marcs, avg3d):
            for parameters in ((5777., 4.44, 0.), (5000., 2.5, -2.), (5300., 3.3, -1.)):
                with self.subTest(grid=module.__name__, parameters=parameters):
                    atmosphere = module.get_model(*parameters, outofgrid_error=True)
                    self.assertGreater(atmosphere['ndepth'], 20)
                    self.assertTrue(np.all(np.diff(atmosphere['lgTauR']) > 0))
                    for key in ('T', 'Pg', 'Pe', 'Density', 'KappaRoss', 'Mu', 'Pturb', 'Prad'):
                        self.assertEqual(len(atmosphere[key]), atmosphere['ndepth'])
                        self.assertTrue(np.all(np.isfinite(atmosphere[key])), key)
                        if key not in ('Pturb', 'Prad'):
                            self.assertTrue(np.all(atmosphere[key] > 0), key)
                    with tempfile.TemporaryDirectory() as directory:
                        filename = Path(directory) / 'atmosphere.mod'
                        atmosphere.write(filename)
                        reader = marcs.read_marcs if module is marcs else avg3d.read_avg3d
                        restored = reader(str(filename))
                    self.assertEqual(atmosphere['ndepth'], restored['ndepth'])
                    for key in ('lgTauR', 'T', 'Pg', 'Pe', 'Density', 'KappaRoss', 'Mu', 'Pturb', 'Prad'):
                        tolerance = {'T': 0.06, 'lgTauR': 0.005, 'Mu': 0.001}.get(key, 1e-12)
                        np.testing.assert_allclose(restored[key], atmosphere[key], rtol=2e-3,
                                                   atol=tolerance, err_msg=key)

    def test_out_of_range_requests_are_rejected(self):
        for module in (marcs, avg3d):
            with self.subTest(grid=module.__name__):
                with self.assertRaisesRegex(ValueError, 'teff out of range'):
                    module.get_model(1000., 4.44, 0., outofgrid_error=True)


if __name__ == '__main__':
    unittest.main()
