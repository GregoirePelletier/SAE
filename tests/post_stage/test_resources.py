"""Tests CPU rapides pour src/post_stage/resources.py (E00, plan §5.4/§16.7).
Aucun GPU, aucune lecture de tenseur -- seulement /proc, cgroup synthetique
et un sous-processus jetable."""
import json
import os
import subprocess
import sys
import time

from src.post_stage.resources import (
    read_proc_status,
    resolve_cgroup_paths,
    read_cgroup_memory,
    gpu_memory_snapshot,
    filesystem_info,
    ResourceSampler,
)


def test_read_proc_status_self_process():
    out = read_proc_status(os.getpid())
    assert "VmRSS" in out
    assert out["VmRSS"] > 0


def test_read_proc_status_missing_pid_returns_empty_not_raise():
    # PID improbable -- ne doit jamais lever, meme si le process n'existe pas.
    out = read_proc_status(2**30)
    assert out == {}


def test_resolve_cgroup_paths_self_does_not_raise():
    info = resolve_cgroup_paths(os.getpid())
    assert "version" in info and "path" in info
    if info["version"] is not None:
        assert os.path.isdir(info["path"])


def test_read_cgroup_memory_v2_synthetic(tmp_path):
    (tmp_path / "memory.current").write_text("104857600\n")
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "memory.peak").write_text("209715200\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    (tmp_path / "memory.stat").write_text(
        "anon 52428800\nfile 10485760\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nother_field 42\n"
    )
    result = read_cgroup_memory({"version": "v2", "path": str(tmp_path)})
    assert result["memory_current"] == 104857600
    assert result["memory_max"] is None  # "max" -> pas de limite -> None, pas une valeur inventee
    assert result["memory_peak"] == 209715200
    assert result["memory_events"]["oom_kill"] == 0
    assert result["memory_stat"]["anon"] == 52428800
    assert "other_field" not in result["memory_stat"]  # seuls les champs d'interet documentes


def test_read_cgroup_memory_missing_path_returns_empty():
    assert read_cgroup_memory({"version": None, "path": None}) == {}


def test_read_cgroup_memory_v1_synthetic(tmp_path):
    (tmp_path / "memory.usage_in_bytes").write_text("52428800\n")
    (tmp_path / "memory.limit_in_bytes").write_text("1073741824\n")
    (tmp_path / "memory.max_usage_in_bytes").write_text("104857600\n")
    (tmp_path / "memory.stat").write_text("cache 1000\nrss 2000\n")
    result = read_cgroup_memory({"version": "v1", "path": str(tmp_path)})
    assert result["memory_current"] == 52428800
    assert result["memory_max"] == 1073741824
    assert result["memory_stat"] == {"cache": 1000, "rss": 2000}


def test_filesystem_info_does_not_raise(tmp_path):
    info = filesystem_info(str(tmp_path))
    assert info["path"] == str(tmp_path)
    assert info["free_bytes"] is None or info["free_bytes"] >= 0


def test_gpu_memory_snapshot_no_gpu_is_fast_and_does_not_raise():
    # Regression : une premiere implementation appelait `import torch` par
    # tick (mesure : ~3,8s sur ce cluster) -- doit rester de l'ordre de
    # quelques centaines de ms au pire (le sous-processus nvidia-smi,
    # absent ici) pour ne jamais dominer un intervalle d'echantillonnage court.
    t0 = time.monotonic()
    result = gpu_memory_snapshot(pid=os.getpid())
    assert time.monotonic() - t0 < 2.0
    assert result == {} or "per_gpu_used_total_mib" in result


def test_resource_sampler_writes_jsonl_and_stops_on_exit(tmp_path):
    out_path = str(tmp_path / "profile.jsonl")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    sampler = ResourceSampler(proc.pid, out_path, interval_s=0.1)
    sampler.set_stage("test_stage")
    sampler.start()
    proc.wait(timeout=10)
    time.sleep(0.3)  # laisse le sampler ecrire le dernier point et sortir de sa boucle
    sampler.stop()

    assert os.path.exists(out_path)
    with open(out_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # Regression : le bug gpu_memory_snapshot ci-dessus faisait sortir la
    # boucle apres UNE seule ligne quel que soit l'intervalle demande --
    # avec 1,0s de duree et un intervalle de 0,1s, plusieurs lignes sont
    # attendues, pas seulement "au moins une".
    assert len(rows) >= 5
    assert rows[0]["stage"] == "test_stage"
    assert "proc_status" in rows[0]
