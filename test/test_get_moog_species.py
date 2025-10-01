from arcane.synthesis import moog
import unittest
import os

class TestVald(unittest.TestCase):
    def test_c2_1(self):
        moogid = moog.get_moog_species_id('C2 1').strip()
        self.assertEqual(moogid,'606.0')
    def test_mg_1(self):
        moogid = moog.get_moog_species_id('Mg 1').strip()
        self.assertEqual(moogid,'12.0')
    def test_mg1(self):
        moogid = moog.get_moog_species_id('Mg1').strip()
        self.assertEqual(moogid,'12.0')
    def test_c2(self):
        moogid = moog.get_moog_species_id('C2').strip()
        self.assertEqual(moogid,'6.1')
    def test_c_2(self):
        moogid = moog.get_moog_species_id('C 2').strip()
        self.assertEqual(moogid,'6.1')
    def test_ch(self):
        moogid = moog.get_moog_species_id('CH').strip()
        self.assertEqual(moogid,'106.0')
    def test_ch_1(self):
        moogid = moog.get_moog_species_id('CH 1').strip()
        self.assertEqual(moogid,'106.0')
    def test_mgi(self):
        moogid = moog.get_moog_species_id('MgI').strip()
        self.assertEqual(moogid,'12.0')
    def test_mgii(self):
        moogid = moog.get_moog_species_id('MgII').strip()
        self.assertEqual(moogid,'12.1')
    def test_mg_i(self):
        moogid = moog.get_moog_species_id('Mg I').strip()
        self.assertEqual(moogid,'12.0')
    def test_mg_ii(self):
        moogid = moog.get_moog_species_id('Mg II').strip()
        self.assertEqual(moogid,'12.1')

if __name__ == '__main__':
    unittest.main()