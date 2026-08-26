"""
pytest configuration for the foundry test suite.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Marker set in the inner pytest invocation so it collects suites normally
# instead of spawning further subprocesses.
_INNER_ENV_VAR = "_FOUNDRY_PYTEST_INNER"

sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")


class SuiteFile(pytest.File):
    """Collector that runs one suite file as a whole in a subprocess."""

    def collect(self):
        yield SuiteItem.from_parent(self, name=self.path.stem, path=self.path)


class SuiteItem(pytest.Item):
    def runtest(self):
        env = os.environ.copy()
        env[_INNER_ENV_VAR] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(self.path),
                "-q",
                "-p",
                "no:logging",
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"{self.path.name} failed: pytest exited with code {result.returncode}"
            )

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, AssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, self.path.name


@pytest.hookimpl(hookwrapper=True)
def pytest_collect_file(file_path, parent):
    """Route `tests_*.py` files to the subprocess suite collector."""
    outcome = yield
    if os.environ.get(_INNER_ENV_VAR):
        return
    # A file requested explicitly on the command line is collected in-process
    # so the user gets the normal per-test output and `-k`/`-x` filtering.
    if parent.session.isinitpath(file_path):
        return
    if (
        file_path.suffix == ".py"
        and file_path.name.startswith("tests_")
        and file_path.parent == TESTS_DIR
    ):
        outcome.force_result([SuiteFile.from_parent(parent, path=file_path)])
