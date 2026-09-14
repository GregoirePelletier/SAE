"""Test boite noire de scripts/post_stage/profile_run.py (E00, plan §16.4) :
verifie que le superviseur retransmet SIGUSR1/SIGTERM a l'enfant qu'il lance
-- sans cela, --signal=USR1@... d'un .slurm n'atteindrait jamais saev5.py une
fois ce superviseur insere entre Slurm et le pipeline (mecanisme de reprise
checkpointee, _GracefulShutdown, cf. saev5.py)."""
import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROFILE_RUN = os.path.join(REPO_ROOT, "scripts", "post_stage", "profile_run.py")


def test_forwards_sigusr1_to_child(tmp_path):
    marker = tmp_path / "marker.txt"
    child_script = (
        "import signal, sys, time\n"
        f"def _h(signum, frame):\n"
        f"    open({str(marker)!r}, 'w').write('got it')\n"
        f"    sys.exit(42)\n"
        "signal.signal(signal.SIGUSR1, _h)\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, PROFILE_RUN,
         "--out", str(tmp_path / "profile.jsonl"),
         "--log", str(tmp_path / "run.log"),
         "--interval", "0.2",
         "--", sys.executable, "-u", "-c", child_script],
    )
    # Laisse le temps au superviseur d'installer ses handlers et de lancer
    # l'enfant avant d'envoyer le signal.
    time.sleep(0.5)
    proc.send_signal(signal.SIGUSR1)
    returncode = proc.wait(timeout=10)

    assert marker.exists(), "l'enfant n'a jamais reçu SIGUSR1 -- signal non retransmis"
    assert returncode == 42, f"code de sortie de l'enfant non propagé (obtenu {returncode})"
