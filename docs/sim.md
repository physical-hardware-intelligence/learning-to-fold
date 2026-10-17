# Simulator runbook (Track S)

Running Ilia's simulation checkpoint in the LeHome Challenge simulator on our GPU box: install, check, evaluate. Every command is the one we ran, with what it does. Gates and pass rules: [plan, Track S](plan.md#track-s--simulation--side-track-started-2026-10-17).

---

## 1. What this is

- **Goal:** run Ilia's *simulation* checkpoint (`lehome_sim`, first of 62 in the simulation round, ~80% success) in the competition's simulator and get his number. That proves we can run his whole system, whatever state our rig is in. Afterwards the simulator is a free test bench for ideas.
- **The simulator** is NVIDIA's **Isaac Sim 5.1**: physics for cloth plus rendering for the cameras. The competition's scene has two SO-101s, a table, a garment and three cameras.
- **One evaluation** = Ilia's `scripts/run_eval.py`. It starts the model server, starts one simulator per worker, and scores every episode with the competition's own keypoint checks.

```
run_eval.py ──starts──► serve.py :8000 (the model, on the GPU)
     │                        ▲  camera images + joints → next moves
     └──starts──► worker ──► Isaac Sim (scene, 2 arms, garment, 3 cameras; renders on the GPU, physics on the CPU)
```

- **Why this GPU:** Isaac Sim 5.1's minimum GPU is the **RTX 4080**; our box has a 4080 SUPER (16 GB). Data-centre cards without RT cores (A100, H100, H200) can't run it.
- **Why one worker:** each simulator needs GPU memory too. Ilia ran 4 on a 96 GB card; we have 16 GB `[estimate]`.

## 2. Where things live

**GPU box** (home folder):

```
~/phi/lehome_solution/                     Ilia's code @ ca73828 (plan, B1) · its .venv/ = the model server env
  outputs/checkpoints/lehome_sim/          the sim checkpoint, 10.5 GB (copied from the laptop)
  lehome-challenge/                        the challenge code @ 5ea947e
    .venv/                                 the simulator env: Isaac Sim 5.1, torch 2.7.0 (made by setup)
    third_party/IsaacLab/                  IsaacLab fork @ 69f6fa5 (made by setup)
    Assets/                                garments, scenes, robots, 0.99 GB (copied from the laptop)
~/phi/learning-to-fold/                    this repo; scripts/sim/ holds the 4 simulator scripts
~/phi/logs/sim/                            setup.log · sim_env_freeze.txt · one folder per run
~/phi/lehome_challenge_uv.lock             the lockfile step 2 installed from
~/.local/bin/zenity                        a do-nothing stand-in (§3, step 1)
~/.local/share/ov/                         Isaac Sim's own caches (196 MB after the first start)
```

**Laptop:** this repo, with the downloads in its git-ignored `checkpoints/` and `data/`.

## 3. One-time setup

```bash
# ---- on the laptop, from this repo's root ----
# 1. download at pinned versions (our box can't reach Hugging Face), then prove every file arrived
export HF_HUB_DISABLE_XET=1                  # Xet stalls on an exFAT drive (plan, trap 18)
hf download IliaLarchenko/lehome_sim --revision 79dcf9e5153c88a56186419d8c71654e50311c15 --local-dir checkpoints/lehome_sim
hf download lehome/asset_challenge --repo-type dataset --revision bea65fd960ad5a1bb3bd3fa77164b28001c08ef9 --local-dir data/lehome_assets
python scripts/verify_hub_download.py fetch IliaLarchenko/lehome_sim 79dcf9e5153c88a56186419d8c71654e50311c15 checkpoints/lehome_sim
python scripts/verify_hub_download.py fetch lehome/asset_challenge bea65fd960ad5a1bb3bd3fa77164b28001c08ef9 data/lehome_assets --repo-type dataset
python scripts/verify_hub_download.py check checkpoints/lehome_sim       # size + hash of every file
python scripts/verify_hub_download.py check data/lehome_assets

# 2. copy to the box (resumable; skips macOS clutter). rsync can't create missing parent folders, so make them first
ssh <you>@<gpu-box> 'mkdir -p phi/lehome_solution/outputs/checkpoints/lehome_sim phi/lehome_solution/lehome-challenge/Assets'
rsync -a --partial --exclude '._*' --exclude '.DS_Store' checkpoints/lehome_sim/ <you>@<gpu-box>:phi/lehome_solution/outputs/checkpoints/lehome_sim/
rsync -a --partial --exclude '._*' --exclude '.DS_Store' data/lehome_assets/   <you>@<gpu-box>:phi/lehome_solution/lehome-challenge/Assets/

# ---- on the box ----
# 3. this repo, next to Ilia's code
git clone https://github.com/physical-hardware-intelligence/learning-to-fold ~/phi/learning-to-fold

# 4. the same proof after the copy (the listing travels with the files, so no internet needed)
python3 ~/phi/learning-to-fold/scripts/verify_hub_download.py check ~/phi/lehome_solution/outputs/checkpoints/lehome_sim
python3 ~/phi/learning-to-fold/scripts/verify_hub_download.py check ~/phi/lehome_solution/lehome-challenge/Assets --allow-extra .gitignore
#   --allow-extra .gitignore = the challenge repo's own tracked file in Assets/

# 5. install the simulator, inside tmux because it takes a few minutes (our box has tmux at ~/miniforge3/bin/tmux)
tmux new -s sim
mkdir -p ~/phi/logs/sim
ACCEPT_ISAACSIM_EULA=yes bash ~/phi/learning-to-fold/scripts/sim/setup_sim_box.sh 2>&1 | tee ~/phi/logs/sim/setup.log
#   ACCEPT_ISAACSIM_EULA=yes = you accept NVIDIA's Isaac Sim license. Without it the script prints the link and installs nothing

# 6. confirm every part is there: 6 isaaclab* packages + lehome
cd ~/phi/lehome_solution/lehome-challenge
~/.local/bin/uv pip list --python .venv/bin/python | grep -i -E "^isaaclab|^lehome"
```

**What `setup_sim_box.sh` does, step by step:**

| step | does | why |
|---|---|---|
| 0 preflight | stops unless the license is accepted; checks the code version, disk, GPU, and that NVIDIA's package site is reachable | fail before downloading 20+ GB |
| 1 zenity | puts a do-nothing `zenity` in `~/.local/bin` | Isaac Sim pops dialogs through `zenity`, which hangs a headless run. Ilia's setup writes it to `/usr/local/bin`, which needs sudo |
| 2 `uv sync` | builds `lehome-challenge/.venv` from the challenge's own lockfile (Isaac Sim 5.1.0, torch 2.7.0) | the same versions every time; the lockfile is kept as `~/phi/lehome_challenge_uv.lock` |
| 3 license | writes Isaac Sim's `EULA_ACCEPTED` file | otherwise it stops to ask |
| 4 IsaacLab | clones the organizers' IsaacLab fork at **`69f6fa5`** | Ilia's setup clones `main` unpinned, so two installs could differ |
| 5 packages | IsaacLab + lehome, plus pins: `setuptools<70`, `flatdict==4.0.1`, `warp-lang==1.12.1` | newer `warp-lang` breaks Isaac Sim 5.1 (Ilia's note) |
| 6 versions | prints what got installed, and saves the full list to `~/phi/logs/sim/sim_env_freeze.txt` | the record: the env no longer matches the lockfile after step 5 |

⚠️ **One error in step 5 is expected.** `isaaclab.sh` fails to build `flatdict 4.0.1` (`No module named 'pkg_resources'`: new setuptools dropped it). The script's next lines install it properly (`setuptools<70`, then `--no-build-isolation`). Ilia's `setup.sh` hits the same error and hides it. Judge the install by command 6, not by that line.

**Why our own script, not `setup.sh`:** Ilia's hides every failure (`2>/dev/null || true`), so a broken install looks fine until the first run. Ours stops at the first error, and it needs no sudo.

**Why `.venv/bin/python`, never `uv run` or `uv sync`, inside `lehome-challenge/`:** both first sync the env back to the lockfile, and step 5 changed it after that. Checked by dry run: `uv run` would change 10 packages (setuptools 69.5.1 → 80.10.1 among them) and add torchaudio; `uv sync` would uninstall 85. Ilia's eval worker calls `.venv/bin/python` directly too. To redo the env, rerun the whole setup script.

## 4. S1 check: does Isaac Sim start?

```bash
# 1. GPU check first, as its own step
bash ~/phi/learning-to-fold/scripts/sim/gpu_guard.sh 12

# 2. start Isaac Sim the way the eval worker does: headless, cameras on, physics on the CPU
cd ~/phi/lehome_solution/lehome-challenge
OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  timeout 1800 .venv/bin/python ~/phi/learning-to-fold/scripts/sim/s1_headless_check.py
#   OMNI_KIT_ACCEPT_EULA=YES = the license answer the worker also gives
#   VK_ICD_FILENAMES        = use NVIDIA's Vulkan driver, as the worker does
#   timeout 1800            = give up after 30 min instead of hanging
#   one line per stage; the last one is "S1 PASS"
```

**A pass looks like this** (2026-10-17): started in 56 s · the challenge's task registers · GPU peak 490 MiB · 60 s in all · nothing left on the GPU.

## 5. Running an evaluation

```bash
# 1. is the GPU free? GO = at least 12 GiB free and nobody else computing on it
bash ~/phi/learning-to-fold/scripts/sim/gpu_guard.sh 12

# 2. first ONE episode, to time it (run inside tmux)
bash ~/phi/learning-to-fold/scripts/sim/run_sim_eval.sh one --num_workers 1 --garment_types top_long --randomize_garments --num_garments 1 --num_episodes 1 --metrics_only --no_wandb
#   the first word ("one") names the run folder; everything after goes to Ilia's run_eval.py

# 3. then the S2 smoke: 2 episodes of every garment (size it from step 2's time per episode)
bash ~/phi/learning-to-fold/scripts/sim/run_sim_eval.sh smoke --num_workers 1 --num_episodes 2 --all --metrics_only --no_wandb

# 4. watch it from another terminal
tail -f ~/phi/logs/sim/*-smoke/eval.log

# 5. done or stopped (Ctrl-C): confirm nothing of ours is left on the GPU
nvidia-smi
```

**`run_eval.py` options that matter:**

| option | default | meaning |
|---|---|---|
| `--all` · `--seen_only` | unseen garments | which garments to test |
| `--garment_types` | all four | any of `top_long top_short pant_long pant_short` |
| `--randomize_garments --num_garments N` | off | test N random garments instead of all |
| `--num_episodes` | 5 | episodes **per garment** |
| `--num_workers` | 4 | simulators at once. **Use 1 on a 16 GB card** |
| `--max_steps` | 600 | 30 s per episode at 20 Hz |
| `--metrics_only` · `--no_wandb` | off | numbers only, no video · no online logging |

**Each run leaves a folder** `~/phi/logs/sim/<date-time>-<name>/`:
- `run.txt`: arguments, the commits of this repo, Ilia's code, the challenge and IsaacLab, start, end, exit code
- `gpu_guard.txt`: the GO decision
- `gpu.csv`: the GPU every 5 s
- `eval.log`: the full output
- `gpu_after.txt`: what was left on the GPU afterwards

## 6. GPU rules (shared box)

- **Check, decide, launch, as separate steps.** `run_sim_eval.sh` runs the guard first and stops on NO-GO.
- **One of our GPU jobs at a time:** this, *or* the robot model server, *or* another experiment. Never two on 16 GB.
- **Long runs only in overnight slots** agreed with the box's owner.
- **Big files stay in your home folder.** On our box `/` holds `/tmp` and had only 21 G free on 2026-10-17.
- Leaving: `nvidia-smi` shows nothing of ours · `tmux ls` holds no finished sessions.

## 7. As performed

### 2026-10-17

**Pins, read before anything ran:**

| what | pin | from |
|---|---|---|
| `IliaLarchenko/lehome_sim` | `79dcf9e` (2026-06-27) · 28 files · 10,501,280,776 B | Hub API |
| `lehome/asset_challenge` | `bea65fd` (2026-03-15) · 447 files · 987,826,911 B | Hub API |
| lehome-challenge | `5ea947e`: Isaac Sim **5.1.0.0**, Python 3.11, torch 2.7.0 (all from its `uv.lock`) | its `pyproject.toml`, `uv.lock` |
| IsaacLab fork | `69f6fa5` (2026-02-04, "debug_headless") | GitHub API |

**Log** (scripts as of commit `4c11670`):

| time (PDT) | step | result |
|---|---|---|
| 09:07 → 09:17 | laptop download + hash check | ✅ checkpoint 28/28, assets 447/447 |
| 09:17 | box preflight | GPU **GO** (14 MiB used) · `pypi.nvidia.com`, PyTorch cu128, PyPI, GitHub all reachable · `/home` 813 G free · `/` 21 G free |
| 09:18:44 → 09:22:24 | **install**, 3 min 40 s | ✅ `uv sync` 2 min 32 s (227 packages from the lockfile) · step 5's expected `flatdict` error, then fixed · Python 3.11.16 · Isaac Sim 5.1.0.0 · IsaacLab 0.53.1 (all 6 packages) · lehome 0.1.0 · torch 2.7.0+cu126 · warp-lang 1.12.1 · flatdict 4.0.1 · numpy 1.26.0 |
| 09:18:53 → | copy laptop → box | ~2.7 MB/s over a relayed connection |
| 09:28:22 | GPU check | **GO**: 14 MiB used, no other jobs |
| 09:28:42 → 09:29:42 | **S1 check** | ✅ **PASS**: started in 55.6 s · torch sees the RTX 4080 SUPER · lehome task registers · GPU peak 490 MiB · nothing left after |
| 09:33 | env recorded | `uv pip freeze`: 293 packages |

**Facts that shaped this:**
- ⚠️ `run_eval.py` **accepts the Isaac Sim license on every start** (`OMNI_KIT_ACCEPT_EULA=YES`). We accepted it explicitly on 2026-10-17, before anything ran.
- Our box **can** reach NVIDIA's and PyTorch's package sites but not Hugging Face, so only the checkpoint and assets go through the laptop.
- The box's `Assets/` held only the repo's tracked `.gitignore`, so the copy overwrites nothing.
- torch is **+cu126, not cu128**: the challenge's `pyproject.toml` lists a cu128 index, but its lockfile takes torch from PyPI. The box's driver (580) runs either.
- The eval worker runs the simulator as `lehome-challenge/.venv/bin/python -m scripts.eval --headless --enable_cameras --device cpu`: **cloth physics on the CPU**, rendering on the GPU.
