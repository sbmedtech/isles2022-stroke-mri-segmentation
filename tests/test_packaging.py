"""Check built-wheel contents and imports outside the source checkout."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile


class PackagingTests(unittest.TestCase):
    def test_wheel_ships_both_namespaces_and_test_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                 '--no-build-isolation', '--wheel-dir', temporary, str(root)],
                check=True, capture_output=True, text=True,
            )
            wheel = next(Path(temporary).glob('isles2022-*.whl'))
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertIn('evaluation/metrics.py', names)
                self.assertIn('evaluation/USAGE.md', names)
                self.assertIn('isles2022/model.py', names)
                metadata = archive.read(next(name for name in names if name.endswith('/METADATA'))).decode()
                self.assertIn('Requires-Dist: numpy>=1.22', metadata)
                self.assertIn('Provides-Extra: test', metadata)
                self.assertIn('Requires-Dist: nibabel; extra == "test"', metadata)
            # -I ignores PYTHONPATH and cwd; verify imports came from this wheel,
            # even if an editable installation is present in the environment.
            probe = '''import sys
sys.path.insert(0, sys.argv[1])
import evaluation, isles2022.config
assert evaluation.__file__.startswith(sys.argv[1])
assert isles2022.config.__file__.startswith(sys.argv[1])
from evaluation import infarct_volume_ml, zero_overlap, bootstrap_patient_ci
'''
            subprocess.run([sys.executable, '-I', '-c', probe, str(wheel)],
                           cwd=temporary, check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
