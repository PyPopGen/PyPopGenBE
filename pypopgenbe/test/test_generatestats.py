import unittest
import numpy as np
from scipy.stats import gmean, scoreatpercentile
from numpy.testing import assert_almost_equal
from pypopgenbe.impl.generatestats import generate_stats


class TestGenerateStats(unittest.TestCase):

    def test_generate_stats_1(self):
        n_organs = 15
        stats = generate_stats(np.zeros((0, n_organs)))

        for key, value in stats.items():
            with self.subTest(key=key):
                assert_almost_equal(value, np.full((1, n_organs), np.nan))

    def test_generate_stats_2(self):
        n_organs = 15
        tissue_data = np.random.rand(1, n_organs)
        stats = generate_stats(tissue_data)
        # tissue_data = tissue_data.reshape(1,15)

        assert_almost_equal(stats['Mean'].reshape(1, 15), tissue_data)
        assert_almost_equal(stats['StdDev'], np.zeros((1, n_organs)))
        assert_almost_equal(stats['GeoMean'].reshape(
            15,), gmean(tissue_data, axis=0))
        assert_almost_equal(stats['GeoStdDev'], np.ones((1, n_organs)))
        assert_almost_equal(stats['P2pt5'].reshape(
            15,), scoreatpercentile(tissue_data, 2.5, axis=0))
        assert_almost_equal(stats['P5'].reshape(
            15,), scoreatpercentile(tissue_data, 5, axis=0))
        assert_almost_equal(stats['Median'].reshape(
            15,), np.median(tissue_data, axis=0))
        assert_almost_equal(stats['P95'].reshape(
            15,), scoreatpercentile(tissue_data, 95, axis=0))
        assert_almost_equal(stats['P97pt5'].reshape(
            15,), scoreatpercentile(tissue_data, 97.5, axis=0))

    def test_generate_stats_3(self):
        n_organs = 15
        tissue_data = np.random.rand(10, n_organs)
        stats = generate_stats(tissue_data)

        assert_almost_equal(stats['Mean'], np.mean(tissue_data, axis=0))
        assert_almost_equal(stats['StdDev'], np.std(
            tissue_data, axis=0, ddof=1))
        assert_almost_equal(stats['GeoMean'], gmean(tissue_data, axis=0))
        assert_almost_equal(stats['GeoStdDev'], np.exp(
            np.std(np.log(tissue_data), axis=0)))

        # Check percentiles only if scipy.stats is available
        if hasattr(np, 'percentile'):
            prc = np.percentile(tissue_data, [2.5, 5, 50, 95, 97.5], axis=0)
            assert_almost_equal(stats['P2pt5'], prc[0, :])
            assert_almost_equal(stats['P5'], prc[1, :])
            assert_almost_equal(stats['Median'], prc[2, :])
            assert_almost_equal(stats['P95'], prc[3, :])
            assert_almost_equal(stats['P97pt5'], prc[4, :])
