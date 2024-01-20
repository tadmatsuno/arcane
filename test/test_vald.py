from arcane_dev.synthesis import readvald
import unittest
import os

class TestVald(unittest.TestCase):
    def test_readvald_stellar_short(self):
        linelist = readvald.read_valdshort('vald/Vald_stellar_short')
        self.assertEqual(len(linelist),54)
    def test_readvald_stellar_short_hfs(self):
        linelist = readvald.read_valdshort('vald/Vald_stellar_short_hfs')
        self.assertEqual(len(linelist),162)
    def test_readvald_all_short(self):
        linelist = readvald.read_valdshort('vald/Vald_all_short')
        self.assertEqual(len(linelist),62934)
    def test_readvald_all_short_hfs(self):
        linelist = readvald.read_valdshort('vald/Vald_all_short_hfs')
        self.assertEqual(len(linelist),63499)

    def test_readvald_stellar_long(self):
        linelist = readvald.read_valdlong('vald/Vald_stellar_long')
        linelist_short = readvald.read_valdshort('vald/Vald_stellar_short')
        for clm in linelist_short.columns:
            if clm != 'references':
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist())

    def test_readvald_stellar_long_hfs(self):
        linelist = readvald.read_valdlong('vald/Vald_stellar_long_hfs')
        linelist_short = readvald.read_valdshort('vald/Vald_stellar_short_hfs')
        for clm in linelist_short.columns:
            if clm != 'references':
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist())

    def test_readvald_all_long(self):
        linelist = readvald.read_valdlong('vald/Vald_all_long')
        linelist_short = readvald.read_valdshort('vald/Vald_all_short')
        for clm in linelist_short.columns:
            if clm != 'references':
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist())

    def test_readvald_all_long_hfs(self):
        linelist = readvald.read_valdlong('vald/Vald_all_long_hfs')
        linelist_short = readvald.read_valdshort('vald/Vald_all_short_hfs')
        for clm in linelist_short.columns:
            print(clm)
            if clm != 'references':
                self.assertEqual(linelist_short[clm].values.tolist(),\
                    linelist[clm].values.tolist())

if __name__ == '__main__':
    unittest.main()