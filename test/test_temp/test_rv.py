import unittest
from arcane.utils import cross_corr
import numpy as np
from astropy.constants import c
ckm = c.to('km/s').value

class TestCrossCorr(unittest.TestCase):
    def test_measure_vshift(self):
        wvl_window = 10.
        wvl_bin = 0.015
        depth = 0.1
        fwhm = 0.1
        sigma = fwhm / (2.*np.sqrt(2.*np.log(2.)))
        noise_level = 0.00
        wc = 5012.
        vshift = 15.5

        xx = np.arange(-wvl_window,wvl_window,wvl_bin)+wc
        fy = lambda x: 1.0-depth * np.exp( - x**2. / (2.*sigma**2.))
        yy = fy(xx-wc)
        yy_obs = fy(xx/(1.0+vshift/ckm)-wc) + \
            noise_level*np.random.randn(len(xx))
        vshift_obs = cross_corr.measure_vshift(xx,yy,yy_obs,max_shift=20.)
        self.assertTrue(np.allclose(vshift,vshift_obs,atol=0.5))
if __name__ == '__main__':
    unittest.main()
