from arcane.synthesis import grid, moog,readvald
from arcane.utils import utils
from arcane.spectrum import model
import numpy as np
import unittest
import os
import iofiles
moog.set_moogsilent_path("/mnt/d/MOOG/mymoog17scat/MOOGSILENT")

valddir = os.path.join(os.path.dirname(__file__),'DATA/vald')
datadir = os.path.join(os.path.dirname(__file__),'DATA')


def fsynth_test(aheight):
    xbins = np.arange(-2,2,0.01)
    yy = aheight*np.exp(- (xbins/0.1)**2/2.)
    return xbins+5000, yy

def fsynth_test2(aheight):
    xbins = np.arange(-2,2,0.01)
    yy = aheight*np.exp(- (xbins/0.1)**2/2.)
    return xbins+5000, 1.0 - yy


class TestLineFitting(unittest.TestCase):
    valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
    sun = iofiles.readspip(os.path.join(datadir, "sun.text"))
    g1 = grid.construct_grid(moog.synth,
        "A_56",
        np.arange(1.5,3.0,0.2),
        linelist = valdlinelist,
        moog_mod_file = os.path.join(datadir,'model.in'),
        workdir='output',no_line_flux_input=-90, wmin=5850, wmax=5860, parallel = True)
    fcontinuum = lambda self,x: 1.0 + (x - 5853.7)*0.05
    fcontinuum2 = lambda self,x: 1.0 + (x - 5000.13)*0.05

    
    def test_simple_height_continuum(self):
        mock_obs_wvl = np.arange(-2,2,0.02) + 5000.13
        _mock_obs_wvl, _mock_obs_flx = fsynth_test2(1.0)
        mock_obs_flx = utils.rebin(_mock_obs_wvl, _mock_obs_flx, mock_obs_wvl, conserve_count=False) + \
            1e-6*np.random.randn(len(mock_obs_wvl)) 
        absmodel = model.LineSynth(
            fsynth_test2,
            synth_parameters={"aheight":0.5},
            parameters_to_fit=["aheight"],
            vfwhm_in=0.0,rv_in=0.0,
            samples = [[5000.13 - 1, 5000.13 + 1]]
            )
        model_continuum = model.ContinuumPolynomial(
            order=1,
            samples = [[5000.13-1.9,5000.13-1],[5000.13+1,5000.13+1.9]]
            )
        res = absmodel.fit(
            mock_obs_wvl, mock_obs_flx* self.fcontinuum2(mock_obs_wvl),
            continuum_model = model_continuum
            )
        self.assertAlmostEqual(res.model_parameters['aheight'],1.0,places=2)
#        self.assertAlmostEqual(res['rv'],0.155,places=2)
#        self.assertAlmostEqual(res['vFWHM'],3.671,places=2)


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
        self.assertAlmostEqual(res.model_parameters['vFWHM'],10.00,places=2)
            
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
        self.assertAlmostEqual(res.model_parameters['vFWHM'],10.00,places=2)

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
            vfwhm_in = 3.65
        )
        wvl_sun = self.sun['wvl'].values
        flx_sun = self.sun['flx'].values
        mask = (wvl_sun>5853.4)&(wvl_sun<5854.0)
        res = model_synth.fit(wvl_sun[mask], flx_sun[mask], optimization_parameter={"diff_step":0.01})
        self.assertAlmostEqual(res.model_parameters['A_56'],2.238,places=2)

    def test_fitting_grid(self):        
        wvl_sun = self.sun['wvl'].values
        flx_sun = self.sun['flx'].values
        mask = (wvl_sun>5853.4)&(wvl_sun<5854.0)
        model_grid = model.LineSynth(
            self.g1,
            synth_parameters={
                "values":2.5
            },
            parameters_to_fit=["values","rv","vFWHM"],
        )
        res = model_grid.fit(wvl_sun[mask], flx_sun[mask])
        self.assertAlmostEqual(res.model_parameters['values'],2.238,places=2)
        self.assertAlmostEqual(res.model_parameters['rv'],0.155,places=2)
        self.assertAlmostEqual(res.model_parameters['vFWHM'],3.65,places=1)

    def test_fitting_grid_depth(self):        
        wvl_sun = self.sun['wvl'].values
        flx_sun = self.sun['flx'].values
        mask = (wvl_sun>5853.4)&(wvl_sun<5854.0)
        model_grid = model.LineSynth(
            self.g1,
            synth_parameters={
                "values":0.1,
                "xaxis":"depth"
            },
            parameters_to_fit=["values","rv","vFWHM"],
        )
        res = model_grid.fit(wvl_sun[mask], flx_sun[mask])
        self.assertAlmostEqual(res.model_parameters['values'],0.628,places=2)
        self.assertAlmostEqual(res.model_parameters['rv'],0.155,places=2)
        self.assertAlmostEqual(res.model_parameters['vFWHM'],3.65,places=1)

    def test_fitting_grid_continuum(self):        
        wvl_sun = self.sun['wvl'].values
        flx_sun = self.sun['flx'].values
        mask = (wvl_sun>5853)&(wvl_sun<5854.4)
        model_line = model.LineSynth(
            self.g1,
            synth_parameters={
                "values":2.5
            },
            parameters_to_fit=["values","rv","vFWHM"],
            samples = [[5853.55, 5853.85]]
        )
        model_continuum = model.ContinuumPolynomial(
            order=1,
            samples = [[5853.0, 5853.4],[5853.9, 5854.4]]
            )
        res = model_line.fit(wvl_sun[mask], flx_sun[mask]*self.fcontinuum(wvl_sun[mask]), 
            continuum_model = model_continuum)
        self.assertAlmostEqual(res.model_parameters['values'],2.22,places=2)
        self.assertAlmostEqual(res.model_parameters['rv'],0.155,places=2)
        self.assertAlmostEqual(res.model_parameters['vFWHM'],3.65,places=1)


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()