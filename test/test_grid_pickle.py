"""Regression coverage for saved grids across SciPy's private-state changes."""

import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from arcane.synthesis.grid import SpectraGrid, load_grid, _LegacyAkima


FIXTURE = Path(__file__).parent / 'fixtures' / 'grid_scipy_1_17_1.pkl'


class TestGridPickle(unittest.TestCase):
    def check_reference(self, grid):
        for name, (query, expected) in grid._reference.items():
            if name in ('values', 'depth', 'ew'):
                actual = grid(query, xaxis=name)[1]
            else:
                actual = getattr(grid, name)(query)
            np.testing.assert_array_equal(actual, expected, err_msg=name)

    def test_legacy_scipy_grid_preserves_all_mappings_without_refitting(self):
        with patch.object(Akima1DInterpolator, '__init__',
                          side_effect=AssertionError('must not refit coefficients')):
            grid = load_grid(FIXTURE)
        self.check_reference(grid)
        self.assertEqual(grid.depth_wavelength, 5002.)

    def test_new_pickle_loads_normally_without_scipy_private_state(self):
        original = load_grid(FIXTURE)
        payload = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertNotIn(b'scipy.', payload)
        with patch.object(Akima1DInterpolator, '__init__',
                          side_effect=AssertionError('must not refit coefficients')):
            restored = pickle.loads(payload)
        self.check_reference(restored)
        for name in ('interpolator', 'depth2flux', 'depth2input', 'input2depth',
                     'ew2flux', 'ew2input', 'input2ew'):
            old, new = getattr(original, name), getattr(restored, name)
            self.assertIs(type(new), Akima1DInterpolator)
            np.testing.assert_array_equal(old.c, new.c)
            np.testing.assert_array_equal(old.x, new.x)
            self.assertEqual(old.extrapolate, new.extrapolate)
            self.assertEqual(old.axis, new.axis)

    def test_loader_accepts_streams_and_current_pickles(self):
        with FIXTURE.open('rb') as stream:
            original = load_grid(stream)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'current.pkl'
            path.write_bytes(pickle.dumps(original))
            self.check_reference(load_grid(path))
        self.check_reference(load_grid(io.BytesIO(pickle.dumps(original))))

    def test_current_scipy_objects_in_older_arcane_pickle(self):
        original = load_grid(FIXTURE)
        # Reproduce caches written before arcane owned interpolation persistence.
        with patch.object(SpectraGrid, '__getstate__', lambda self: self.__dict__):
            payload = pickle.dumps(original)
        self.check_reference(load_grid(io.BytesIO(payload)))

    def test_older_slot_layout_preserves_axis_and_extrapolation(self):
        x = np.linspace(0, 1, 6)
        original = Akima1DInterpolator(x, np.array([x**2, x**3]),
                                       axis=1, extrapolate=False)
        legacy = _LegacyAkima()
        legacy.__setstate__((None, dict(c=original.c, x=original.x,
                                       extrapolate=False, axis=1)))
        recovered = legacy.restore()
        query = np.array([-.1, .1, .5, .9, 1.1])
        np.testing.assert_array_equal(recovered(query), original(query))
        self.assertEqual(recovered.axis, 1)
        self.assertFalse(recovered.extrapolate)

    def test_current_multidimensional_grid_round_trip(self):
        values = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        grid = SpectraGrid(['a', 'b'], values, values + 1, np.array([5000., 5001.]))
        restored = load_grid(io.BytesIO(pickle.dumps(grid)))
        query = [[.2, .5], [.6, .1]]
        np.testing.assert_array_equal(restored(query)[1], grid(query)[1])

    def test_wrong_pickle_type_is_reported(self):
        with self.assertRaisesRegex(TypeError, 'SpectraGrid'):
            load_grid(io.BytesIO(pickle.dumps(np.arange(3))))


if __name__ == '__main__':
    unittest.main()
