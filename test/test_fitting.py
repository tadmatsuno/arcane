from arcane.synthesis import grid, moog,readvald
from arcane.utils import utils
from arcane.spectrum import model
import numpy as np
import unittest
import os
import iofiles

valddir = os.path.join(os.path.dirname(__file__),'DATA/vald')
datadir = os.path.join(os.path.dirname(__file__),'DATA')


def fsynth_test(aheight):
    xbins = np.arange(-2,2,0.01)
    yy = aheight*np.exp(- (xbins/0.1)**2/2.)
    return xbins+5000, yy

class TestLineFitting(unittest.TestCase):
    valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
    sun = iofiles.readspip(os.path.join(datadir, "sun.text"))

    def test_simple_heigt(self):
        mock_obs_wvl = np.arange(-1.5,1.5,0.02) + 5000.13
        _mock_obs_wvl, _mock_obs_flx = fsynth_test(1.0)
        mock_obs_flx = utils.rebin(_mock_obs_wvl, _mock_obs_flx, mock_obs_wvl, conserve_count=False) + \
            1e-6*np.random.randn(len(mock_obs_wvl))
        absmodel = model.LineSynth(
            fsynth_test,
            synth_parameters={"aheight":0.5},
            parameters_to_fit=["aheight"],
            vfwhm_in=0.0,rv_in=0.0)
        res = absmodel.fit(mock_obs_wvl,mock_obs_flx)
        self.assertAlmostEqual(res.model_parameters['aheight'],1.0,places=2)
        
    def test_simple_heigtrv(self):
        mock_obs_wvl = np.arange(-1.5,1.5,0.02) + 5000.13
        _mock_obs_wvl, _mock_obs_flx = fsynth_test(1.0)
        mock_obs_flx = utils.rebin(_mock_obs_wvl, _mock_obs_flx, mock_obs_wvl, conserve_count=False) + \
            1e-6*np.random.randn(len(mock_obs_wvl))
        absmodel = model.LineSynth(
            fsynth_test,
            synth_parameters={"aheight":0.5},
            parameters_to_fit=["aheight","rv"],
            vfwhm_in=0.0,rv_in=1.0)
        res = absmodel.fit(mock_obs_wvl,mock_obs_flx)
        self.assertAlmostEqual(res.model_parameters['aheight'],1.0,places=2)
        self.assertAlmostEqual(res.model_parameters['rv'],0.0,places=2)
            
    def test_simple_heigtvfwhm(self):
        mock_obs_wvl = np.arange(-1.5,1.5,0.02) + 5000.13
        _mock_obs_wvl, _mock_obs_flx = fsynth_test(1.0)
        _mock_obs_flx = utils.smooth_spectrum(_mock_obs_wvl,_mock_obs_flx,10)
        mock_obs_flx = utils.rebin(_mock_obs_wvl, _mock_obs_flx, mock_obs_wvl, conserve_count=False) + \
            1e-6*np.random.randn(len(mock_obs_wvl))
        absmodel = model.LineSynth(
            fsynth_test,
            synth_parameters={"aheight":0.5},
            parameters_to_fit=["aheight","vFWHM"],
            vfwhm_in=5.,rv_in=0.0)
        res = absmodel.fit(mock_obs_wvl,mock_obs_flx)
        self.assertAlmostEqual(res.model_parameters['aheight'],1.0,places=2)
        self.assertAlmostEqual(res.model_parameters['vFWHM'],10.01,places=2)
            
    def test_simple_heigtrvvfwhm(self):
        mock_obs_wvl = np.arange(-1.5,1.5,0.02) + 5000.13
        _mock_obs_wvl, _mock_obs_flx = fsynth_test(1.0)
        _mock_obs_flx = utils.smooth_spectrum(_mock_obs_wvl,_mock_obs_flx,10)
        mock_obs_flx = utils.rebin(_mock_obs_wvl, _mock_obs_flx, mock_obs_wvl, conserve_count=False) + \
            1e-6*np.random.randn(len(mock_obs_wvl))
        absmodel = model.LineSynth(
            fsynth_test,
            synth_parameters={"aheight":0.5},
            parameters_to_fit=["aheight","rv","vFWHM"],
            vfwhm_in=5.,rv_in=1.0)
        res = absmodel.fit(mock_obs_wvl,mock_obs_flx)
        self.assertAlmostEqual(res.model_parameters['aheight'],1.0,places=2)
        self.assertAlmostEqual(res.model_parameters['rv'],0.0,places=2)
        self.assertAlmostEqual(res.model_parameters['vFWHM'],10.01,places=2)

    def test_fitting_moog(self):
        model_synth = model.LineSynth(
            moog.synth,
            synth_parameters={
                "linelist":self.valdlinelist,
                "teff":5777,
                "logg":4.44,
                "vt":1.0,
                "feh":0.0,
                "workdir":"output",
                "A_56":2.5
            },
            parameters_to_fit=["A_56"],
            rv_in = 0.155,
            vfwhm_in = 3.661
        )
        wvl_sun = self.sun['wvl'].values
        flx_sun = self.sun['flx'].values
        mask = (wvl_sun>5853.4)&(wvl_sun<5854.0)
        res = model_synth.fit(wvl_sun[mask], flx_sun[mask])
        self.assertAlmostEqual(res.model_parameters['A_56'],2.238,places=2)

    def test_fitting_grid(self):        

        g1 = grid.construct_grid(moog.synth,
            "A_56",
            np.arange(0.0,3.0,0.2),
            linelist = self.valdlinelist,
            moog_mod_file = os.path.join(datadir,'model.in'),
            workdir='output',no_line_flux_input=-90, wmin=5850, wmax=5860, parallel = True)
        wvl_sun = self.sun['wvl'].values
        flx_sun = self.sun['flx'].values
        mask = (wvl_sun>5853.4)&(wvl_sun<5854.0)
        model_grid = model.LineSynth(
            g1,
            synth_parameters={
                "values":2.5
            },
            parameters_to_fit=["values","rv","vFWHM"],
        )
        res = model_grid.fit(wvl_sun[mask], flx_sun[mask])
        self.assertAlmostEqual(res.model_parameters['values'],2.238,places=2)
        self.assertAlmostEqual(res.model_parameters['rv'],0.155,places=2)
        self.assertAlmostEqual(res.model_parameters['vFWHM'],3.661,places=2)

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()