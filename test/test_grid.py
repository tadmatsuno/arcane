import unittest
import os 
from arcane.synthesis import moog,grid,readvald
import numpy as np

valddir = os.path.join(os.path.dirname(__file__),'DATA/vald')
datadir = os.path.join(os.path.dirname(__file__),'DATA')

class TestGrid(unittest.TestCase):
    valdlinelist = readvald.readvald(os.path.join(valddir,'Vald_stellar_short_hfs'))
    moog5603 = moog.synth(\
        valdlinelist,
        run_id='test_moog_moogmod',
        moog_mod_file=os.path.join(datadir,'model.in'),
        workdir='output', wmin=5850, wmax=5860,A_56 = 0.3)
    moog5600 = moog.synth(\
        valdlinelist,
        run_id='test_moog_moogmod',
        moog_mod_file=os.path.join(datadir,'model.in'),
        workdir='output', wmin=5850, wmax=5860,A_56 = -90)
    depth = np.max(moog5600[1] - moog5603[1])
    def test_grid(self): 
        g1 = grid.construct_grid(moog.synth,
            "A_56",
            np.arange(0.0,1.0,0.2),
            linelist = self.valdlinelist,
            run_id='test_moog_moogmod',
            moog_mod_file=os.path.join(datadir,'model.in'),
            workdir='output',no_line_flux_input=-90, wmin=5850, wmax=5860, parallel = True)
        self.assertAlmostEqual(np.max(np.abs(g1(0.3)-self.moog5603[1])), 0.0, places=3)
        self.assertAlmostEqual(self.depth, g1.input2depth(0.3), places=3)
        self.assertAlmostEqual(np.max(np.abs(g1.depth2flux(self.depth)-self.moog5603[1])), 0.0, places=3)
        
    def test_grid2(self): 
        g1 = grid.construct_grid(moog.synth,
            "AX_dict",
            [{56:v} for v in np.arange(0.0,1.0,0.2)],
            grid_values = lambda d: d[56],
            linelist = self.valdlinelist,
            run_id='test_moog_moogmod',
            moog_mod_file=os.path.join(datadir,'model.in'),
            workdir='output', wmin=5850, wmax=5860, parallel = True)
        self.assertAlmostEqual(np.max(np.abs(g1(0.3)-self.moog5603[1])), 0.0, places=3)



        
if __name__ == '__main__':
    unittest.main()
        