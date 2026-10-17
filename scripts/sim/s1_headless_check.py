"""Track S, gate S1: does Isaac Sim 5.1 start headless here, the way the challenge starts it?

Run with the challenge's own env (on the GPU box, after gpu_guard.sh says GO):
    cd ~/phi/lehome_solution/lehome-challenge
    OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        timeout 1800 .venv/bin/python ~/phi/learning-to-fold/scripts/sim/s1_headless_check.py

Same settings as Ilia's eval worker (--headless --enable_cameras --device cpu). One line per
stage with elapsed seconds; "S1 PASS" once the app has started, run and closed. The first start
is slow: Isaac Sim builds its shader cache.
"""
import importlib.metadata as md
import time

t0 = time.time()


def stage(msg: str) -> None:
    print(f"[s1 {time.time() - t0:7.1f}s] {msg}", flush=True)


stage("isaacsim " + md.version("isaacsim") + " · isaaclab " + md.version("isaaclab"))
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True, device="cpu").app
stage("Isaac Sim started: headless, cameras on")

import torch  # noqa: E402

stage(f"torch {torch.__version__} · CUDA {torch.cuda.is_available()} · "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
try:  # what the challenge's eval.py imports next; informational, not part of the gate
    import lehome.tasks.bedroom  # noqa: F401

    stage("lehome task registered")
except Exception as e:  # noqa: BLE001
    stage(f"lehome task import failed (not the S1 gate): {type(e).__name__}: {e}")
for _ in range(60):
    app.update()
stage("60 app updates")
stage("S1 PASS")
app.close()
