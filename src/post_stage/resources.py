"""
src/post_stage/resources.py -- Instrumentation memoire/IO pour l'audit E00
(docs/post_stage/PLAN_E00-E09.md §5.4).

Aucune dependance sur saev5.py : ce module observe un PROCESSUS EXTERNE (pid
d'un job lance en subprocess) via /proc et cgroup, sans jamais importer ni
modifier le pipeline d'entrainement lui-meme (§16.1 -- adaptateurs, pas de
reecriture du monolithe).

Distinctions a ne pas confondre (§5.1) : RAM physique du noeud, memoire
facturee au cgroup du job, RSS du processus, espace virtuel mappe, cache de
fichier, memoire partagee IPC, memoire GPU -- chaque fonction ci-dessous ne
lit qu'UNE de ces quantites et la nomme explicitement, jamais un scalaire
agrege qui les confondrait.
"""

import os
import time
import json
import subprocess
import threading
from typing import Optional


def read_proc_status(pid: int) -> dict:
    """VmRSS/VmHWM/RssAnon/RssFile/RssShmem en octets, depuis /proc/<pid>/status.
    Retourne un dict partiel si le process a disparu entre l'appel et la
    lecture (jamais d'exception -- un sampler en tache de fond ne doit pas
    mourir sur une lecture manquee)."""
    out: dict = {}
    path = f"/proc/{pid}/status"
    try:
        with open(path) as f:
            for line in f:
                for key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
                    if line.startswith(key + ":"):
                        # Format "Key:\t   1234 kB"
                        parts = line.split()
                        if len(parts) >= 2 and parts[-1] == "kB":
                            out[key] = int(parts[-2]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return out


def resolve_cgroup_paths(pid: int) -> dict:
    """Resout le VRAI chemin cgroup du processus depuis /proc/<pid>/cgroup --
    ne JAMAIS supposer un chemin fixe (le plan §5.4 l'exige explicitement :
    le chemin differe entre session interactive et job Slurm, et entre sites).
    v2 (unifie, ligne "0::<path>") : un seul chemin sous /sys/fs/cgroup.
    v1 (une ligne par controleur) : chemin specifique au controleur "memory"
    sous /sys/fs/cgroup/memory.
    Retourne {"version": "v2"|"v1"|None, "path": str|None}."""
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            lines = [l.strip() for l in f if l.strip()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {"version": None, "path": None}

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hier_id, controllers, rel_path = parts
        if hier_id == "0" and controllers == "":
            # cgroup v2 unifie
            abs_path = os.path.join("/sys/fs/cgroup", rel_path.lstrip("/"))
            if os.path.isdir(abs_path):
                return {"version": "v2", "path": abs_path}
        elif "memory" in controllers.split(","):
            abs_path = os.path.join("/sys/fs/cgroup/memory", rel_path.lstrip("/"))
            if os.path.isdir(abs_path):
                return {"version": "v1", "path": abs_path}
    return {"version": None, "path": None}


def _read_kv_file(path: str) -> dict:
    """Parse un fichier cgroup 'key value' par ligne (memory.stat, memory.events).
    Absence de fichier -> dict vide (ex. memory.peak n'existe qu'a partir du
    noyau 5.19 -- ne jamais supposer sa presence)."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                bits = line.split()
                if len(bits) == 2:
                    try:
                        out[bits[0]] = int(bits[1])
                    except ValueError:
                        out[bits[0]] = bits[1]
    except (FileNotFoundError, PermissionError):
        pass
    return out


def _read_scalar_file(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            val = f.read().strip()
        return None if val == "max" else int(val)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


_STAT_FIELDS_OF_INTEREST = ("anon", "file", "file_mapped", "file_dirty", "file_writeback")


def read_cgroup_memory(cgroup_info: dict) -> dict:
    """Lit memory.current/memory.max (+ memory.peak si present) et les
    champs de memory.stat/memory.events utiles a distinguer memoire anonyme
    (le vrai risque OOM) de cache fichier (recuperable par le noyau sous
    pression, cf. §5.1/§5.7). Repli v1 (memory.usage_in_bytes/limit_in_bytes,
    memory.stat sans les memes cles) plutot que de supposer v2 partout."""
    version, path = cgroup_info.get("version"), cgroup_info.get("path")
    if path is None:
        return {}
    if version == "v2":
        stat = _read_kv_file(os.path.join(path, "memory.stat"))
        return {
            "memory_current": _read_scalar_file(os.path.join(path, "memory.current")),
            "memory_max": _read_scalar_file(os.path.join(path, "memory.max")),
            "memory_peak": _read_scalar_file(os.path.join(path, "memory.peak")),
            "memory_events": _read_kv_file(os.path.join(path, "memory.events")),
            "memory_stat": {k: stat[k] for k in _STAT_FIELDS_OF_INTEREST if k in stat},
        }
    if version == "v1":
        stat = _read_kv_file(os.path.join(path, "memory.stat"))
        return {
            "memory_current": _read_scalar_file(os.path.join(path, "memory.usage_in_bytes")),
            "memory_max": _read_scalar_file(os.path.join(path, "memory.limit_in_bytes")),
            "memory_peak": _read_scalar_file(os.path.join(path, "memory.max_usage_in_bytes")),
            "memory_events": {},  # pas d'equivalent direct memory.events sous v1
            "memory_stat": {k: stat[k] for k in ("cache", "rss") if k in stat},
        }
    return {}


def gpu_memory_snapshot(pid: Optional[int] = None, timeout_s: float = 5.0) -> dict:
    """Memoire GPU via `nvidia-smi` (process EXTERNE au sampler -- ne PAS
    utiliser `torch.cuda.memory_allocated()` ici : ce module observe un pid
    cible different du sien, et `import torch` a lui seul coute plusieurs
    secondes sur ce cluster (mesure : ~3,8s sur le noeud frontal, cf.
    preflight) -- un cout inacceptable a payer a CHAQUE tick du sampler).
    Absence de `nvidia-smi` (noeud sans GPU, ex. frontal) -> dict vide, sans
    lever. `pid` filtre `memory.used` au processus cible via
    --query-compute-apps ; sans pid, seul le total par carte est rapporte."""
    out: dict = {"per_gpu_used_total_mib": {}, "per_pid_used_mib": {}}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                idx, used, total = (p.strip() for p in line.split(","))
                out["per_gpu_used_total_mib"][idx] = {"used": int(used), "total": int(total)}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return {}

    if pid is not None:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=timeout_s,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    app_pid, used = (p.strip() for p in line.split(","))
                    if int(app_pid) == pid:
                        out["per_pid_used_mib"][str(pid)] = int(used)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
    return out if out["per_gpu_used_total_mib"] else {}


def filesystem_info(path: str) -> dict:
    """Type de filesystem (depuis /proc/mounts, plus long prefixe de montage
    correspondant) et espace libre/total (os.statvfs) -- §5.4 : un memmap sur
    reseau lent a un profil different d'un SSD local, a savoir avant de
    profiler, pas a supposer."""
    path = os.path.abspath(path)
    fstype = None
    best_len = -1
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, mount_fstype = parts[1], parts[2]
                if path.startswith(mount_point) and len(mount_point) > best_len:
                    best_len, fstype = len(mount_point), mount_fstype
    except FileNotFoundError:
        pass
    try:
        st = os.statvfs(path)
        free_bytes = st.f_bavail * st.f_frsize
        total_bytes = st.f_blocks * st.f_frsize
    except OSError:
        free_bytes = total_bytes = None
    return {"path": path, "fstype": fstype, "free_bytes": free_bytes, "total_bytes": total_bytes}


class ResourceSampler:
    """Echantillonneur en tache de fond (thread daemon) : ecrit une ligne
    JSONL par tick dans `out_path`, en ecriture ajout + flush (jamais de
    troncature, jamais perdu si le process surveille est tue avant l'arret
    propre du sampler). `set_stage`/`set_progress` sont appeles par
    l'appelant (ex. un parseur des logs "[timing] ..." du pipeline
    surveille) pour etiqueter chaque ligne sans que ce module ait besoin de
    connaitre la structure interne du pipeline."""

    def __init__(self, pid: int, out_path: str, interval_s: float = 20.0):
        self.pid = pid
        self.out_path = out_path
        self.interval_s = interval_s
        self._stage = "unknown"
        self._progress: dict = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.monotonic()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    def set_stage(self, name: str) -> None:
        self._stage = name

    def set_progress(self, **kwargs) -> None:
        self._progress.update(kwargs)

    def _sample_once(self) -> dict:
        cgroup_info = resolve_cgroup_paths(self.pid)
        return {
            "ts": time.time(),
            "elapsed_s": round(time.monotonic() - self._t0, 1),
            "pid": self.pid,
            "stage": self._stage,
            "progress": dict(self._progress),
            "proc_status": read_proc_status(self.pid),
            "cgroup_version": cgroup_info.get("version"),
            "cgroup": read_cgroup_memory(cgroup_info),
            "gpu": gpu_memory_snapshot(pid=self.pid),
        }

    def _loop(self) -> None:
        with open(self.out_path, "a") as f:
            while not self._stop.is_set():
                row = self._sample_once()
                f.write(json.dumps(row) + "\n")
                f.flush()
                if not read_proc_status(self.pid):
                    # Process disparu : dernier point ecrit, plus rien a suivre.
                    break
                self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
