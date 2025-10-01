
from arcane.synthesis import turbospectrum as ts
from arcane.utils import utils
import pandas
import matplotlib.pyplot as plt
import iofiles
import numpy as np
import unittest
from arcane.synthesis import readvald
from arcane.mdlatm import marcs
import os

vald_stellar = readvald.read_valdshort('./vald//Vald_stellar_short')

#ts.write_linelist(vald_stellar,"./output/test_test.lines")

hd122563 = iofiles.readspip('./DATA/HD122563plsp.op')


vald_stellar_short = readvald.readvald('./vald//Vald_stellar_short')

common_inputs = {"teff":4636,"logg":1.418,"vt":2.05,"feh":-2.60,"A_6":5.220,"A_56":-1.80,
                 "workdir":"output","wmin":5851, "wmax":5858}

import importlib
importlib.reload(ts)

class TestTurbospectrum(unittest.TestCase):
    def test_linelist_turbospectrum(self):
        wvl0, flx0 = ts.synth(linelist=vald_stellar_short, 
            run_id='test_ts_vald_short',
            return_cntm = False, **common_inputs)
               
        for linelist, run_id in \
            zip([vald_stellar_short.to_dict(orient="list"), \
                "output/linelist_test_ts_vald_short1750949083.029.lin",
                ["output/linelist_test_ts_vald_short1750949083.029.lin",os.path.join(ts.DATA_path, "Hlinedata")],
                ],
                ['test_ts_vald_short_dict', "test_ts_file","whydrogen"]):
            wvl1, flx1 = ts.synth(linelist=linelist,
                run_id=run_id, return_cntm = False, **common_inputs)
            flx1 = utils.rebin(wvl1, flx1, wvl0, conserve_count=False)
            print(f"Comparing {run_id} with test_ts_vald_short")
            self.assertTrue(np.allclose(flx0, flx1, atol=1.0e-3))
            
if __name__ == '__main__':
    unittest.main()
