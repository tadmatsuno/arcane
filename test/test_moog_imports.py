"""MOOG utilities should work without optional atmosphere grids."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMoogImports(unittest.TestCase):
    def test_import_without_configuration_or_atmosphere_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_config = str(Path(directory) / 'missing.arcanesrc')
            code = f'''
import os.path
import sys
from unittest.mock import patch
expanduser = os.path.expanduser
def test_expanduser(path):
    return {missing_config!r} if path == '~/.arcanesrc' else expanduser(path)
with patch('os.path.expanduser', side_effect=test_expanduser):
    from arcane.synthesis import moog
assert 'arcane.mdlatm.marcs' not in sys.modules
assert 'arcane.mdlatm.avg3d' not in sys.modules
assert moog.get_moog_species_id('Fe I') == '26.0'
'''
            result = subprocess.run(
                [sys.executable, '-c', code], capture_output=True, text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
