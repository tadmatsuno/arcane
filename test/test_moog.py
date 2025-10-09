from arcane.synthesis import moog,readvald
from arcane.utils import utils
import unittest
import numpy as np
import os

valddir = os.path.join(os.path.dirname(__file__),'DATA/vald')
datadir = os.path.join(os.path.dirname(__file__),'DATA')

class TestMoog(unittest.TestCase):
    def test_moog_pdlinelist(self):
        valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = moog.synth(\
            linelist=valdlinelist,
            run_id='test_moog_pdll',
            teff=5777, logg=4.44, vt=1.0, feh=0.0,
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(1.0-flx[np.argmin(np.abs(wvl-5853.67))],0.2989,places=2)
    
    def test_moog_moogmod(self):
        valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = moog.synth(\
            linelist=valdlinelist,
            run_id='test_moog_moogmod',
            moog_mod_file=os.path.join(datadir,'model.in'),
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(1.0-flx[np.argmin(np.abs(wvl-5853.67))],0.2989,places=2)

    def test_moog_mooglin(self):
        wvl, flx = moog.synth(\
            linelist=os.path.join(datadir,"lines_synth.in"),
            run_id='test_moog_mooglin',
            moog_mod_file=os.path.join(datadir,'model.in'),
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(1.0-flx[np.argmin(np.abs(wvl-5853.67))],0.2989,places=2)
    
    def test_moog_marcsmod(self):
        valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        wvl, flx = moog.synth(\
            linelist=valdlinelist,
            run_id='test_moog_marcsmod',
            marcs_mod_file=os.path.join(datadir,'sun_marcs.mod'),vt=1.0,
            workdir='output')
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertAlmostEqual(1.0-flx[np.argmin(np.abs(wvl-5853.67))],0.2989,places=2)

    def test_moog_abundance_input1(self):
        wvl, flx = moog.synth(\
            linelist=os.path.join(datadir,"lines_synth.in"),
            run_id='test_moog_abundance_input',
            moog_mod_file=os.path.join(datadir,'model.in'),
            workdir='output',A_56=2.5)
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertTrue(1.0-flx[np.argmin(np.abs(wvl-5853.67))]<0.2989)
    
    def test_moog_abundance_input2(self):
        wvl, flx = moog.synth(\
            linelist=os.path.join(datadir,"lines_synth.in"),
            run_id='test_moog_abundance_input',
            moog_mod_file=os.path.join(datadir,'model.in'),
            workdir='output',AX_dict={56:2.5})
        self.assertTrue(wvl[0]<5850)
        self.assertTrue(wvl[-1]>5860)
        self.assertTrue(1.0-flx[np.argmin(np.abs(wvl-5853.67))]<0.2989)

if __name__ == '__main__':
    unittest.main()
