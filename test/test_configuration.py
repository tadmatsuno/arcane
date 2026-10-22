"""A fresh Python installation must not depend on external program setup."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arcane.config import read_config, set_path


class TestConfiguration(unittest.TestCase):
    def test_first_setting_and_backup_preserve_other_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            config = str(Path(directory) / '.arcanesrc')
            resource = Path(directory) / 'external'
            resource.touch()
            self.assertEqual(read_config(config), {})
            self.assertFalse(Path(config).exists())
            set_path(config, 'moog_silent', resource)
            first = Path(config).read_text()
            set_path(config, 'marcs_dir', directory)
            self.assertEqual(Path(config + '_old').read_text(), first)
            self.assertEqual(read_config(config), {
                'moog_silent': str(resource), 'marcs_dir': directory})
            with self.assertRaises(FileNotFoundError):
                set_path(config, 'avg3d_dir', Path(directory) / 'missing')
            self.assertNotIn('avg3d_dir', read_config(config))

    def test_missing_and_partial_configurations_allow_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / '.arcanesrc'
            # Test each in a fresh interpreter; the real user configuration is untouched.
            for settings in (None, {}, {'moog_silent': '/missing/MOOGSILENT'}):
                if settings is not None:
                    config.write_text(json.dumps(settings))
                code = f'''
import os.path
from unittest.mock import patch
expanduser = os.path.expanduser
def expand(path):
    if path == '~/.arcanesrc':
        return {str(config)!r}
    if path == '~':
        return {directory!r}
    return expanduser(path)
with patch('os.path.expanduser', side_effect=expand):
    from arcane.mdlatm import marcs, avg3d
    from arcane.synthesis import moog, turbospectrum
for call, key in [
    (lambda: marcs.get_marcs_mod(5777, 4.44, 0), 'marcs_dir'),
    (lambda: avg3d.get_model(5777, 4.44, 0), 'avg3d_dir'),
    (lambda: moog.run_moog('syn', None), 'moog_silent'),
    (lambda: turbospectrum.run_turbospectrum('syn'), 'turbospectrum_root'),
]:
    try:
        call()
    except RuntimeError as error:
        assert key in str(error), error
    else:
        raise AssertionError('Expected configuration error for ' + key)
'''
                result = subprocess.run([sys.executable, '-c', code],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if settings is None:
                    self.assertFalse(config.exists())

    def test_grid_setters_reload_data(self):
        from arcane.mdlatm import marcs, avg3d
        with tempfile.TemporaryDirectory() as directory:
            config = str(Path(directory) / '.arcanesrc')
            Path(directory, 'MARCS_avai.dat').write_text('teff      5000 6000\n')
            Path(directory, 'MARCS_grid.csv').write_text('teff,filename\n5000,model.mod\n')
            with patch.object(marcs, 'src_path', config), \
                    patch.object(marcs, 'grid'), patch.object(marcs, 'grid_value'), \
                    patch.object(marcs, 'data_dir'):
                marcs.set_marcs_path(directory)
                self.assertEqual(marcs.grid['teff'].tolist(), [5000])
                self.assertEqual(marcs.grid_value['teff'].tolist(), [5000, 6000])
            with patch.object(avg3d, 'src_path', config), \
                    patch.object(avg3d, 'data_dir'), patch.object(avg3d, 'allmodels'), \
                    patch.object(avg3d, 'readsav', return_value={'a': 'new models'}):
                avg3d.set_avg3d_path(directory)
                self.assertEqual(avg3d.allmodels, 'new models')


if __name__ == '__main__':
    unittest.main()
