from arcane.synthesis import moog
from arcane.utils import utils
import pandas
import matplotlib.pyplot as plt
import iofiles
import numpy as np
import unittest
from arcane.synthesis import readvald
from arcane.mdlatm import marcs


hd122563 = iofiles.readspip('./DATA/HD122563plsp.op')

linelist_csv = pandas.read_csv('./vald/stellar_short.csv',dtype={'moog_species':str})
linelist_csv.to_dict(orient="list")

vald_stellar_short = readvald.readvald('./vald//Vald_stellar_short')
vald_stellar_long = readvald.readvald('./vald/Vald_stellar_long')
vald_stellar_short_hfs = readvald.readvald('./vald//Vald_stellar_short_hfs')
vald_stellar_long_hfs = readvald.readvald('./vald/Vald_stellar_long_hfs')

vald_wide = readvald.readvald('./vald/240122Valdlong_3600_7000_solar0.001_hfssolar.txt')

common_inputs = {"teff":4636,"logg":1.418,"vt":2.05,"feh":-2.60,"A_6":5.220,"A_56":-1.80,
                 "workdir":"output","wmin":5851, "wmax":5858}
class TestLinelist(unittest.TestCase):
    def test_linelist_moog(self):
    # Test if different linelist formats give more or less the same results
        wvl0, flx0 = moog.synth(\
            linelist=linelist_csv, run_id='test_moog_csv',#,scat=0,
            **common_inputs
            )
        for linelist, run_id in \
            zip([vald_stellar_short, vald_stellar_long, \
                    vald_stellar_short_hfs, vald_stellar_long_hfs, \
                    linelist_csv.to_dict(orient="list"),"output/line_test_moog_csv_dict.in"],
                ['test_moog_vald_short', 'test_moog_vald_long',\
                    'test_moog_vald_short_hfs', 'test_moog_vald_long_hfs', 
                    'test_moog_csv_dict', "test_moog_file"]):
            wvl1,flx1 = moog.synth(linelist=linelist,
                run_id=run_id, **common_inputs)
            flx1 = utils.rebin(wvl1, flx1, wvl0, conserve_count=False)
            print(f"Comparing {run_id} with test_moog_csv")
            self.assertTrue(np.allclose(flx0, flx1,atol=1.0e-3))
        
    def test_param_moog(self):
        # Test if different parameters give more or less the same results
        wvl0, flx0 = moog.synth(\
            linelist=linelist_csv, run_id='test_moog_param',
            **common_inputs
            )
        
        print("Testing marcs .mod file input")
        model_file = marcs.get_marcs_mod(\
            teff=common_inputs["teff"],
            logg=common_inputs["logg"],
            mh=common_inputs["feh"]).write(\
                "./output/marcsHD122563.mod")
        wvl1, flx1 = moog.synth(linelist=linelist_csv,
            marcs_mod_file = "./output/marcsHD122563.mod",
            run_id='test_moog_marcs_mod',
            vt =  common_inputs["vt"],
            A_6 = common_inputs["A_6"],
            A_56 = common_inputs["A_56"],
            workdir = common_inputs["workdir"],
            wmin=common_inputs["wmin"], wmax=common_inputs["wmax"])
        self.assertTrue(np.allclose(flx0, flx1,atol=1.0e-3))
        
        print("Testing moog .in file input")
        wvl1, flx1 = moog.synth(linelist=linelist_csv,
            moog_mod_file = "output/model_test_moog_param.in",
            run_id='test_moog_marcs_mod',
            A_6 = common_inputs["A_6"],
            A_56 = common_inputs["A_56"],
            workdir = common_inputs["workdir"],
            wmin=common_inputs["wmin"], wmax=common_inputs["wmax"])
        
        print("A warnings is expected here.")
        wvl1, flx1 = moog.synth(linelist=linelist_csv,
            moog_mod_file = "output/model_test_moog_param.in",
            run_id='test_moog_marcs_mod',
            feh = common_inputs["feh"],
            A_6 = common_inputs["A_6"],
            A_56 = common_inputs["A_56"],
            workdir = common_inputs["workdir"],
            wmin=common_inputs["wmin"], wmax=common_inputs["wmax"])
        
        print("A warning is expected here.")
        wvl1, flx1 = moog.synth(linelist=linelist_csv,
            moog_mod_file = "output/model_test_moog_param.in",
            run_id='test_moog_marcs_mod',
            vt = common_inputs["vt"],
            A_6 = common_inputs["A_6"],
            A_56 = common_inputs["A_56"],
            workdir = common_inputs["workdir"],
            wmin=common_inputs["wmin"], wmax=common_inputs["wmax"])

    def test_abund_moog(self):
        # Test if different abundance inputs yield the same results
        wvl0, flx0 = moog.synth(\
            linelist=linelist_csv, run_id='test_moog_param',
            **common_inputs
            )
        
        abundances = {6: common_inputs["A_6"],
                      56: common_inputs["A_56"]}
        
        wvl1, flx1 = moog.synth(linelist=linelist_csv,
            AX_dict = abundances,
            teff =common_inputs["teff"],
            logg = common_inputs["logg"],
            vt = common_inputs["vt"],
            feh = common_inputs["feh"],
            run_id='test_moog_abund',
            workdir = common_inputs["workdir"],
            wmin=common_inputs["wmin"], wmax=common_inputs["wmax"])
        self.assertTrue(np.allclose(flx0, flx1,atol=1.0e-3))

if __name__ == '__main__':
    unittest.main()