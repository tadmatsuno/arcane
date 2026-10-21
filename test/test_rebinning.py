"""Rebinning against independently integrated pixel overlaps."""
import unittest

import numpy as np

from arcane.utils import utils


def pixel_edges(centers):
    return np.r_[centers[0]-(centers[1]-centers[0])/2.,
                 (centers[1:]+centers[:-1])/2.,
                 centers[-1]+(centers[-1]-centers[-2])/2.]


def overlap_integrals(source, flux, target, counts):
    left = pixel_edges(source)
    right = pixel_edges(target)
    density = flux/np.diff(left) if counts else flux
    result = []
    for low, high in zip(right[:-1], right[1:]):
        overlap = np.maximum(0., np.minimum(left[1:], high)-np.maximum(left[:-1], low))
        value = np.sum(overlap*density)
        result.append(value if counts else value/(high-low))
    return np.asarray(result)


class TestRebinning(unittest.TestCase):
    def test_irregular_grids_match_integrated_pixel_overlaps(self):
        rng = np.random.RandomState(31)
        source = np.cumsum(rng.uniform(0.05, 0.15, 100))
        target = np.linspace(source[5], source[-6], 47)
        for counts in (False, True):
            for missing in (False, True):
                with self.subTest(counts=counts, missing=missing):
                    flux = rng.uniform(0.1, 2., len(source))
                    if missing:
                        flux[[20, 45, 75]] = np.nan
                    mask = np.isfinite(flux)
                    expected = overlap_integrals(source[mask], flux[mask], target, counts)
                    actual = utils.rebin(source, flux, target, conserve_count=counts)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_gaps_remain_zero_and_uncovered_chunks_are_skipped(self):
        source = np.r_[np.arange(0., 2., 0.1), np.arange(5., 7., 0.1)]
        target = np.arange(0.2, 6.7, 0.1)
        actual = utils.rebin(source, np.ones(len(source)), target,
                             conserve_count=False, npix_mask_criterion=3.)
        np.testing.assert_array_equal(actual[(target > 2.1) & (target < 4.9)], 0.)
        # Output windows need not intersect every source segment/chunk.
        target = np.arange(5.2, 6.5, 0.1)
        actual = utils.rebin(source, np.ones(len(source)), target,
                             conserve_count=False, npix_mask_criterion=3.)
        np.testing.assert_allclose(actual, 1., atol=1e-12)

    def test_nonoverlapping_output_is_zero(self):
        actual = utils.rebin(np.arange(20.), np.ones(20), np.arange(30., 40.))
        np.testing.assert_array_equal(actual, np.zeros(10))


if __name__ == '__main__':
    unittest.main()
