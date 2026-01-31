import unittest
import os
from arcane.utils import solarabundance as sa
from arcane.utils import isotopic_ratio as ir

class TestSolarAbundance(unittest.TestCase):
    def test_get_atomnum(self):
        self.assertEqual(sa.get_atomnum('H'),1)
        self.assertEqual(sa.get_atomnum('He'),2)
        self.assertEqual(sa.get_atomnum('Fe'),26)
    
    def test_get_elemname(self):
        self.assertEqual(sa.get_elemname(1),'H')
        self.assertEqual(sa.get_elemname(2),'He')
        self.assertEqual(sa.get_elemname(26),'Fe')
    
    def test_get_solar(self):
        self.assertAlmostEqual(sa.get_solar('H'),12.00)
        self.assertAlmostEqual(sa.get_solar('He'),10.914)
        self.assertAlmostEqual(sa.get_solar('Fe'),7.46)
        self.assertAlmostEqual(sa.get_solar(1),12.00)
        self.assertAlmostEqual(sa.get_solar(2),10.914)
        self.assertAlmostEqual(sa.get_solar(26),7.46)
        
    def test_isotopic_ratio(self):
        frac_C12 = ir.get_fraction('C',12)
        frac_C13 = ir.get_fraction('C',13)
        self.assertAlmostEqual(frac_C12,0.98893)
        self.assertAlmostEqual(frac_C13,0.01107)
        frac_C12 = ir.get_fraction(6,12)
        frac_C13 = ir.get_fraction(6,13)
        self.assertAlmostEqual(frac_C12,0.98893)
        self.assertAlmostEqual(frac_C13,0.01107)
        fractions = ir.get_fraction('C',[12,13])
        self.assertAlmostEqual(fractions[0],0.98893)
        self.assertAlmostEqual(fractions[1],0.01107)
        fractions = ir.get_fraction(6,[12,13])
        self.assertAlmostEqual(fractions[0],0.98893)
        self.assertAlmostEqual(fractions[1],0.01107)
        fractions = ir.get_fraction(['Ca','Ti'],48)
        self.assertAlmostEqual(fractions[0],0.00187)
        self.assertAlmostEqual(fractions[1],0.7372)
        

if __name__ == '__main__':
    sa.load_solarabundance("Asplund2021")
    ir.load_isotopic_ratio("solar_Asplund2021")
    unittest.main()