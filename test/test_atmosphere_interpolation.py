"""Small model structures exercise interpolation edge cases."""
import copy
import unittest

import numpy as np

from arcane.mdlatm.marcs import MARCS, interp_model2


def atmosphere(depths, offset=0.):
    depths = np.asarray(depths, dtype=float)
    return MARCS(
        modeltype='marcs', geometry='plane-parallel', modelname='test',
        comment='', ndepth=len(depths), lgTauR=depths,
        T=5000. + offset + depths * 100., Pg=10.**(depths + 4. + offset / 1000.),
        Pturb=np.zeros(len(depths)),
    )


class TestAtmosphereInterpolation(unittest.TestCase):
    def test_partial_overlap_resamples_both_inputs_without_mutation(self):
        first = atmosphere([-3, -2, -1, 0, 1])
        second = atmosphere([-2, -1, 0, 1, 2], offset=100.)
        originals = copy.deepcopy([first, second])
        result = interp_model2(first, second, 0.5)
        np.testing.assert_array_equal(result['lgTauR'], [-2, -1, 0, 1])
        np.testing.assert_allclose(result['T'], [4850., 4950., 5050., 5150.])
        np.testing.assert_allclose(result['Pg'], 10.**(result['lgTauR'] + 4.05))
        for actual, original in zip((first, second), originals):
            for key in original:
                np.testing.assert_equal(actual[key], original[key], err_msg=key)

    def test_resampling_does_not_modify_wider_second_input(self):
        first = atmosphere([-2, -1, 0, 1])
        second = atmosphere([-3, -2, -1, 0, 1, 2])
        original = copy.deepcopy(second)
        interp_model2(first, second, 0.5)
        for key in original:
            np.testing.assert_equal(second[key], original[key], err_msg=key)

    def test_zero_pressure_at_interpolation_endpoints_stays_finite(self):
        first = atmosphere([-2, -1, 0, 1])
        second = atmosphere([-2, -1, 0, 1], offset=100.)
        second['Pturb'][:] = 2.
        for weight in (0., 0.5, 1.):
            with self.subTest(weight=weight):
                with np.errstate(divide='raise', invalid='raise'):
                    result = interp_model2(first, second, weight)
                self.assertTrue(np.all(np.isfinite(result['Pturb'])))
                expected = second['Pturb'] if weight == 1. else first['Pturb']
                np.testing.assert_array_equal(result['Pturb'], expected)

    def test_disjoint_depth_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'common.*depth'):
            interp_model2(atmosphere([-4, -3, -2]), atmosphere([0, 1, 2]), 0.5)


if __name__ == '__main__':
    unittest.main()
