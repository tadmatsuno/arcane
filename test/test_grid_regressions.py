"""Small analytic synthesis grids, independent of external programs."""
import unittest

import numpy as np

from arcane.synthesis.grid import SpectraGrid


class TestGridRegressions(unittest.TestCase):
    def make_grid(self, depths, **kwargs):
        return SpectraGrid(
            ['abundance'], np.arange(len(depths), dtype=float)[:, None],
            1. - np.asarray(depths)[:, None] * np.array([[0.2, 1., 0.2]]),
            np.array([5000., 5001., 5002.]), no_line_flux=np.ones(3), **kwargs,
        )

    def test_depth_mapping_discards_points_below_previous_maximum(self):
        grid = self.make_grid([0.1, 0.4, 0.2, 0.3, 0.5])
        np.testing.assert_allclose(grid.depth2input([0.1, 0.4, 0.5]), [0., 1., 4.], atol=1e-12)
        np.testing.assert_allclose(grid(0.4, xaxis='depth')[1], [0.92, 0.6, 0.92])
        np.testing.assert_allclose(grid(-0.4, xaxis='depth')[1], [1.08, 1.4, 1.08])

    def test_empty_depth_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'depth_wr_range'):
            self.make_grid([0.1, 0.2, 0.3], depth_wr_range=(6000., 6001.))

    def test_depth_mapping_requires_two_positive_increasing_samples(self):
        for depths in ([0., 0., 0.], [0.2, 0.1, 0.1]):
            with self.subTest(depths=depths):
                with self.assertRaisesRegex(ValueError, 'two.*depth'):
                    self.make_grid(depths)


if __name__ == '__main__':
    unittest.main()
