"""Run all test scripts in the tests/ folder."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).parent

SCRIPTS = [
    "tests_distributed.py",
    "tests_gym.py",
    "tests_synthetic.py",
]

# Redirect all .pyc compilation to a temp directory
env = os.environ.copy()
env["PYTHONPYCACHEPREFIX"] = os.path.join(tempfile.gettempdir(), "pycache")

failed = []
for script in SCRIPTS:
    path = TESTS_DIR / script
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)
    result = subprocess.run([sys.executable, str(path)], env=env)
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}", file=sys.stderr)
        failed.append(script)

print(f"\n{'='*60}")
if failed:
    print(f"FAILED: {', '.join(failed)}")
    sys.exit(1)
else:
    print("All tests passed 🎉🎉🎉")
