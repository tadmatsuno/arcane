from arcane.synthesis import moog
import unittest
import os

class TestVald(unittest.TestCase):
    def test_all(self):
        moogid = moog.get_moog_species_id('C2 1').strip()
        self.assertEqual(moogid,'606.0')

        moogid = moog.get_moog_species_id('Mg 1').strip()
        self.assertEqual(moogid,'12.0')

        moogid = moog.get_moog_species_id('Mg1').strip()
        self.assertEqual(moogid,'12.0')

        moogid = moog.get_moog_species_id('C2').strip()
        self.assertEqual(moogid,'6.1')

        moogid = moog.get_moog_species_id('C 2').strip()
        self.assertEqual(moogid,'6.1')

        moogid = moog.get_moog_species_id('CH').strip()
        self.assertEqual(moogid,'106.0')

        moogid = moog.get_moog_species_id('CH 1').strip()
        self.assertEqual(moogid,'106.0')

        moogid = moog.get_moog_species_id('MgI').strip()
        self.assertEqual(moogid,'12.0')

        moogid = moog.get_moog_species_id('MgII').strip()
        self.assertEqual(moogid,'12.1')

        moogid = moog.get_moog_species_id('Mg I').strip()
        self.assertEqual(moogid,'12.0')

        moogid = moog.get_moog_species_id('Mg II').strip()
        self.assertEqual(moogid,'12.1')

if __name__ == '__main__':
    unittest.main()