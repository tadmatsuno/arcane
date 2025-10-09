from arcane.synthesis import turbospectrum,readvald
from arcane.utils import utils
import unittest
import numpy as np
import os

valddir = os.path.join(os.path.dirname(__file__),'DATA/vald')
datadir = os.path.join(os.path.dirname(__file__),'DATA')

class TestTS(unittest.TestCase):
    def test_ts_pdlinelist(self):
        valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = turbospectrum.synth(\
            linelist=valdlinelist,
            run_id='test_ts_pdll',
            teff=5777, logg=4.44, vt=1.0, feh=0.0,
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(flx[np.argmin(np.abs(wvl-5853.67))],0.29698,places=2)

    def test_ts_tslinelist(self):
        #valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = turbospectrum.synth(\
            linelist=os.path.join(datadir,"lines_ts.in"),
            run_id='test_ts_tslinelist',
            teff=5777, logg=4.44, vt=1.0, feh=0.0,
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(flx[np.argmin(np.abs(wvl-5853.67))],0.29698,places=2)

    def test_ts_tslinelistlist(self):
        #valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = turbospectrum.synth(\
            linelist=[os.path.join(datadir,"lines_ts.in")],
            run_id='test_ts_tslinelistlist',
            teff=5777, logg=4.44, vt=1.0, feh=0.0,
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(flx[np.argmin(np.abs(wvl-5853.67))],0.29698,places=2)

    def test_ts_opac(self):
        valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = turbospectrum.synth(\
            linelist=valdlinelist,
            run_id='test_ts_opac',
            ts_opac_file=os.path.join(datadir,'opac.in'),
            workdir='output', spherical=False, feh=0.0)
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(flx[np.argmin(np.abs(wvl-5853.67))],0.29698,places=2)
        
    def test_ts_marcs(self):
        valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = turbospectrum.synth(\
            linelist=valdlinelist,
            run_id='test_ts_marcs',
            marcs_mod_file=os.path.join(datadir,'sun_marcs.mod'),
            workdir='output',vt=1.,feh=0.0)
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(flx[np.argmin(np.abs(wvl-5853.67))],0.29698,places=2)

    def test_moog_abundance_input1(self):
        wvl, flx = turbospectrum.synth(\
            linelist=os.path.join(datadir,"lines_ts.in"),
            run_id='test_ts_abundance_input1',
            marcs_mod_file=os.path.join(datadir,'sun_marcs.mod'),
            workdir='output',A_56=2.5,vt=1.,feh=0.0)
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertTrue(flx[np.argmin(np.abs(wvl-5853.67))]<0.29698)
    
    def test_moog_abundance_input2(self):
        wvl, flx = turbospectrum.synth(\
            linelist=os.path.join(datadir,"lines_ts.in"),
            run_id='test_ts_abundance_input2',
            marcs_mod_file=os.path.join(datadir,'sun_marcs.mod'),
            workdir='output',AX_dict={56:2.5},vt=1.,feh=0.0)
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertTrue(flx[np.argmin(np.abs(wvl-5853.67))]<0.29698)

if __name__ == '__main__':
    unittest.main()
