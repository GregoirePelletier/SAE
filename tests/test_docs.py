import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_check_docs_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_docs.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
