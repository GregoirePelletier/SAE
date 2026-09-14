"""
scripts/post_stage/profile_run.py -- Superviseur de profilage E00
(Plan_execution_SAE_15_jours_Claude_Code.md §5.4/§5.7).

Lance la pipeline reelle (ex. `.venv/bin/python src/sae/saev5.py`) en
subprocess et echantillonne sa memoire/cgroup/GPU pendant qu'elle tourne, SANS
modifier saev5.py -- l'etiquette de "stage" est deduite en tailant son stdout
a la recherche des marqueurs `stage_timer` deja presents dans le code
("[timing] X..." / "[timing] X termine en ...s"), jamais d'un import du
pipeline. Usage :

    .venv/bin/python scripts/post_stage/profile_run.py \\
        --out resources_profile.jsonl --log run.log --interval 20 \\
        -- .venv/bin/python -u src/sae/saev5.py

Tout ce qui suit `--` est la commande a lancer telle quelle (env deja
positionne par l'appelant, typiquement un .slurm).
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.post_stage.resources import ResourceSampler, filesystem_info  # noqa: E402

_STAGE_START_RE = re.compile(r"\[timing\]\s+(.*?)\.\.\.\s*$")
_STAGE_END_RE = re.compile(r"\[timing\]\s+(.*?)\s+termin[ée]\s+en\s+([\d.]+)s")


def _tail_stdout(proc: subprocess.Popen, sampler: ResourceSampler, log_path: str) -> None:
    with open(log_path, "w") as logf:
        for raw_line in proc.stdout:
            logf.write(raw_line)
            logf.flush()
            m_start = _STAGE_START_RE.search(raw_line)
            m_end = _STAGE_END_RE.search(raw_line)
            if m_end:
                sampler.set_progress(last_stage_duration_s=float(m_end.group(2)))
                sampler.set_stage(f"post:{m_end.group(1)}")
            elif m_start:
                sampler.set_stage(m_start.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Chemin resources_profile.jsonl")
    ap.add_argument("--log", required=True, help="Chemin du stdout+stderr complet du job surveille")
    ap.add_argument("--interval", type=float, default=20.0, help="Secondes entre deux echantillons")
    ap.add_argument("--fs-check-path", default=".", help="Chemin dont on rapporte le filesystem/quota")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Commande a lancer après --")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("commande manquante après --")

    fs_info = filesystem_info(args.fs_check_path)
    print(f"[profile_run] filesystem cible : {fs_info}", flush=True)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # Propagation du signal de checkpoint (§16.4 : ce superviseur s'insere
    # entre Slurm et le pipeline reel -- sans ceci, --signal=USR1@... du
    # .slurm atteindrait CE process, pas saev5.py, et son mecanisme de
    # reprise checkpointee (_GracefulShutdown) ne se declencherait jamais).
    # Re-emis vers l'enfant DIRECT (pas de process group ici, un seul
    # niveau de subprocess) ; SIGTERM/SIGINT transmis pour le meme motif
    # (arret propre demande par Slurm sur depassement de --time, ou Ctrl-C
    # interactif).
    def _forward_signal(signum, _frame):
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    for _sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _forward_signal)

    sampler = ResourceSampler(proc.pid, args.out, interval_s=args.interval)
    sampler.set_stage("startup")
    sampler.start()

    tail_thread = threading.Thread(target=_tail_stdout, args=(proc, sampler, args.log), daemon=True)
    tail_thread.start()

    t0 = time.monotonic()
    returncode = proc.wait()
    tail_thread.join(timeout=10)
    sampler.set_stage("exited")
    sampler.stop()

    summary = {
        "cmd": cmd, "returncode": returncode,
        "wall_seconds": round(time.monotonic() - t0, 1),
        "filesystem": fs_info,
    }
    summary_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "profile_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[profile_run] termine, code={returncode}, resume ecrit dans {summary_path}", flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
