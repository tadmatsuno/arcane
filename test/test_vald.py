from arcane.synthesis import readvald
import unittest
import os

valddir = os.path.join(os.path.dirname(__file__),'DATA/vald')

class TestVald(unittest.TestCase):
    def test_readvald_stellar_short(self):
        linelist = readvald.read_vald(os.path.join(valddir,'Vald_stellar_short'))
        self.assertEqual(len(linelist),54)

    def test_readvald_stellar_short_hfs(self):
        linelist = readvald.read_vald(os.path.join(valddir,'Vald_stellar_short_hfs'))
        self.assertEqual(len(linelist),162)

    def test_readvald_stellar_long(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_stellar_long'))
        self.assertEqual(len(linelist),54)

    def test_readvald_stellar_long_hfs(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_stellar_long_hfs'))
        self.assertEqual(len(linelist),162)

    def test_readvald_all_short(self):
        linelist = readvald.read_valdshort(os.path.join(valddir,'Vald_all_short'))
        self.assertEqual(len(linelist),62934)

    def test_readvald_all_short_hfs(self):
        linelist = readvald.read_valdshort(os.path.join(valddir,'Vald_all_short_hfs'))
        self.assertEqual(len(linelist),63499)

    def test_readvald_all_long(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_all_long'))
        self.assertEqual(len(linelist),62934)

    def test_readvald_all_long_hfs(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_all_long_hfs'))
        self.assertEqual(len(linelist),63499)

    def test_readvald_stellar_long2(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_stellar_long'))
        linelist_short = readvald.read_valdshort(os.path.join(valddir,'Vald_stellar_short'))
        for clm in linelist_short.columns:
            if not clm in ['references','vturb']:
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist(),msg="stellar_long - stellar short, "+clm)

    def test_readvald_stellar_long_hfs2(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_stellar_long_hfs'))
        linelist_short = readvald.read_valdshort(os.path.join(valddir,'Vald_stellar_short_hfs'))
        for clm in linelist_short.columns:
            if not clm in ['references','vturb']:
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist(),msg="stellar_long_hfs - stellar short_hfs, "+clm)

    def test_readvald_all_long2(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_all_long'))
        linelist_short = readvald.read_valdshort(os.path.join(valddir,'Vald_all_short'))
        for clm in linelist_short.columns:
            if not clm in ['references','vturb']:
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist(),msg="all_long - all short, "+clm)

    def test_readvald_all_long_hfs2(self):
        linelist = readvald.read_valdlong(os.path.join(valddir,'Vald_all_long_hfs'))
        linelist_short = readvald.read_valdshort(os.path.join(valddir,'Vald_all_short_hfs'))
        for clm in linelist_short.columns:
            if not clm in ['references','vturb']:
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist(),msg="all_long_hfs - all_short_hfs, "+clm)

    def test_scaling(self):
        scaled = readvald.readvald(os.path.join(valddir,'Vald_stellar_long_hfs'))
        noscaled = readvald.readvald(os.path.join(valddir,'Vald_stellar_long_hfs_noscaling'))
        self.assertTrue(scaled.scaled)
        self.assertFalse(noscaled.scaled)
        noscaled.get_scaling()
        noscaled.apply_scaling()
        self.assertTrue(noscaled.scaled)
        self.assertAlmostEqual(\
            scaled[scaled["species"]=="Ba 2"].loggf.values[0],
            noscaled[noscaled["species"]=="Ba 2"].loggf.values[0], places=2)
                

if __name__ == '__main__':
    unittest.main()
