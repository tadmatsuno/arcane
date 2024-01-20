from arcane_dev.mdlatm import marcs,avg3d
import unittest
import os


class TestModelAtm(unittest.TestCase):
    def test_MARCS(self):
        """Test MARCS model atmosphere class."""
        # Test MARCS model atmosphere class
        sun = marcs.get_marcs_mod(5777,4.44,0.0,0.0)
        self.assertIsNotNone(sun)
        sun.write('output/sun_marcs.mod')
        self.assertTrue(os.path.exists('output/sun_marcs.mod'))

    def test_avg3D(self):
        '''
        Test time averaged 3D model atmospheres.
        '''
        sun = avg3d.get_model(5777,4.44,0.0,0.0)
        self.assertIsNotNone(sun)
        #sun.write('output/sun_avg3d.mod')
        #self.assertTrue(os.path.exists('output/sun_avg3d.mod'))

if __name__ == '__main__':
    unittest.main()