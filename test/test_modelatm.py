from arcane.mdlatm import marcs,avg3d
import unittest
import os
from outputdir import check_temp_dir, clear_temp_dir


class TestModelAtm(unittest.TestCase):
    def test_MARCS(self):
        """Test MARCS model atmosphere class."""
        # Test MARCS model atmosphere class
        sun = marcs.get_marcs_mod(5777,4.44,0.0,0.0)
        self.assertIsNotNone(sun)
        sun.write('output/sun_marcs.mod')
        self.assertTrue(os.path.exists('output/sun_marcs.mod'))
        sun2 = marcs.read_marcs('output/sun_marcs.mod')
        self.assertEqual(sun['ndepth'], sun2['ndepth'])
        self.assertAlmostEqual(sun['T'][0], sun2['T'][0], places=1)

    def test_avg3D(self):
        '''
        Test time averaged 3D model atmospheres.
        '''
        sun = avg3d.get_model(5777,4.44,0.0,0.0)
        self.assertIsNotNone(sun) 
        sun.write('output/sun_avg3d.mod')
        self.assertTrue(os.path.exists('output/sun_avg3d.mod'))
        sun2 = avg3d.read_avg3d('output/sun_avg3d.mod')
        self.assertEqual(sun['ndepth'], sun2['ndepth'])
        self.assertAlmostEqual(sun['T'][0], sun2['T'][0], places=1)
        
    

if __name__ == '__main__':
    unittest.main()