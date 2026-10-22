"""GUI modules must import using only declared package dependencies."""

import os
import subprocess
import sys
import unittest


class TestGuiImports(unittest.TestCase):
    def test_gui_imports_and_equivalent_width_table(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        code = '''
from PyQt5.QtWidgets import QApplication
app = QApplication([])
from arcane.continuum import continuum
from arcane.ew import linesfit
assert linesfit.ptable[26] == 'Fe'
assert linesfit.iofiles.__name__ == 'arcane.utils.iofiles'
'''
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
