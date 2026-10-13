"""Build and import the wheel outside the checkout, without network access."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


class PackagingTests(unittest.TestCase):
    def test_wheel_contains_both_namespaces_and_test_dependency(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([
                sys.executable, "-m", "pip", "wheel", str(root), "--no-deps",
                "--no-build-isolation", "--no-index", "--wheel-dir", directory,
            ], check=True, capture_output=True, text=True)
            wheel = next(Path(directory).glob("isles2022-*.whl"))
            target = Path(directory) / "installed"
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertIn("evaluation/metrics.py", names)
                self.assertIn("evaluation/USAGE.md", names)
                self.assertIn("isles2022/model.py", names)
                self.assertIn("isles2022/reproducibility.py", names)
                modules = (
                    "checkpoints", "config", "data", "datasets", "evaluation",
                    "preprocessing", "provenance", "training", "visualization",
                )
                for module in modules:
                    self.assertIn(f"isles2022/{module}.py", names)
                metadata = archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
                self.assertIn('Requires-Dist: nibabel; extra == "test"', metadata)
                self.assertIn('Requires-Dist: numpy>=1.22', metadata)
                self.assertIn('Provides-Extra: notebook', metadata)
                archive.extractall(target)
            subprocess.run([
                sys.executable, "-I", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import evaluation, isles2022.reproducibility, importlib; "
                "assert evaluation.__file__.startswith(sys.argv[1]); "
                "assert isles2022.reproducibility.__file__.startswith(sys.argv[1]); "
                "modules = [importlib.import_module('isles2022.' + name) "
                "for name in sys.argv[2:]]; "
                "assert all(module.__file__.startswith(sys.argv[1]) for module in modules)",
                str(target),
                *modules,
            ], cwd=directory, check=True, capture_output=True, text=True)
