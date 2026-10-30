# Running Ilia's `lehome_real` on our SO-101s: everything we changed

Ilia Larchenko's 2nd-place LeHome real-round checkpoint, run **zero-shot** (no training on our data) on Φ's bimanual SO-101 rig. First fold: 2026-10-29. Hand towel, square and centred: **5 of 5 folded in half** (runs 3, 4, 7, 8, 9), plus a sixth seen by eye (run 10, no video). Other layouts: 0 of 2. Boxer shorts: 0 of 2, before the camera fixes below.

This page lists every change between "download the checkpoint" and "the towel folds", in the order a new rig needs them. The model itself is explained in [the explainer](https://physical-hardware-intelligence.github.io/blog/learning-to-fold-end-to-end.html) and in Ilia's [tech report](https://arxiv.org/abs/2606.27163); a short summary is at the end.

## 1. Rig geometry

| What | Ours | Why |
|---|---|---|
| Arms | two SO-101 followers, pan axes **46 cm apart**, two leaders | the organisers' rig spacing |
| Top camera | EMEET SmartCam C960, centred between the arms, **73.5 cm** above the table, tilted a few degrees | as close to the organisers' view as we could get (§3) |
| Table | **white** sheet filling the top camera's view | the organisers' table is white; our black Squint cloth was the biggest visible difference |
| Wrist cameras | icspring UVC on each fixed jaw, **horn side** | the right one needed a corrected mount (TheRobotStudio's one-piece part puts the camera on the motor side, 180° from the left arm's) |
| Arm colour | white (theirs are orange) | tested offline 2026-10-22: repainting their arms white in their frames made the model's errors ~7% worse, so we kept white |

## 2. Arms: units, calibration, ports

- **Units: degrees end to end** (`use_degrees=True`: arm joints in degrees, gripper 0–100). Ilia lost days to LeRobot's change from the −100…100 encoding to degrees (report §9.11); his released code is degree mode only.
- **Calibration:** each arm keeps its own LeRobot calibration (`phi_bi_follower_left/right`, `phi_bi_leader_left/right`). We applied **no** joint offsets to match the organisers'. Their rest poses vs ours: elbow end stops within ~1°, left shoulder lift ~3° further back (one reading).
- **Ports:** the leaders were crossed in our local config (left/right swapped) until 2026-10-18. Check each arm's identity read-only before a session: the stored calibration in its motors must match its file.
- **A replaced motor** on the right follower (2026-10-17) had to be set to its ID and the arm recalibrated.

## 3. Cameras

| # | Change | Why |
|---|---|---|
| 1 | **Top crop:** keep the centre **79.4%** of the frame, scale it to 1280×720 (`--top_crop 0.794`) | our lens is wider than the organisers' D435 colour camera (spec 82° vs 69° wide). The crop makes ours frame like theirs at our height. The recording keeps the full frame |
| 2 | **Aim the top camera by overlay** (`real2sim2real/experiments/E007_overhead_camera/align.py --crop 0.794`): their frame laid over our live view, move the camera until the arm bases sit on theirs | Ilia's own most useful tool was the same idea (report §9.7). ~80% overlay was enough |
| 3 | **Top camera at 1280×720, not 1920×1080** | three cameras at once on one USB hub: at 1080p a wrist camera gave no frame within 1 s. The model shrinks every image to 224 px wide, so nothing is lost |
| 4 | **Both wrist images rotated 180°** (`rotation: 180` per camera in the config) | the organisers' wrist views show the fingers at the bottom, pointing up; ours were upside down. A rotation, not a mirror: the fixed jaw sits where theirs does |
| 5 | **Focus** the new wrist camera by hand (`real2sim2real/experiments/E006_wrist_lens/focus.py`) | it arrived blurred: sharpness 0.9 → 123 |
| 6 | **White balance auto on both wrist cameras** (uvcc, by USB address: both have the same vendor/product id) | one camera kept a 4000 K lock from Squint and gave a pink picture. Squint's camera lock sets it again: recheck before folding |
| 7 | Top camera exposure stays **auto** | the C960 refuses UVC control requests on macOS; on a white table auto was fine |
| 8 | **Camera numbers change with USB order** | find them every session (`overhead.py which`) and update the local config |

## 4. Safety (each found on our rig before it hurt anything)

| # | Change | What would have happened |
|---|---|---|
| 1 | **Goal = present before torque** (`_guard_configure`) | LeRobot 0.6.0's follower connect turns torque on without setting the goal. After a power-up our right follower's stored goals were ~270° from its pose on five joints: it would have jerked there. Leaders get stale goals too |
| 2 | **Safe stop**: every exit ramps all 12 joints back to the pose read at connect, holds until each is within 2°, then torque off | upstream turns torque off wherever the arms are (its rest ramp is never called, and its rest pose has the elbow at 100°, past our stop) |
| 3 | `--leaders_limp` | in autonomous mode upstream powers the leaders and snaps them to the followers' pose (up to ~90° on our left leader's roll) |
| 4 | `--start_delay_s 5` countdown | one run started while a hand was still placing the towel |
| 5 | `--dry_run` first on every new setup | everything runs, nothing moves (followers limp, leaders never torqued) |
| 6 | **No speed cap** for real runs | a 2° per step cap (`--max_relative_target 2`) made the arms sluggish and the model's moves were clamped almost every step. Ilia runs without one |

## 5. Speed and plumbing

| # | Change | Effect |
|---|---|---|
| 1 | websocket **compression off** | 0.68 → 0.28 s per answer on the GPU box (2026-10-03) |
| 2 | `--client_resize`: frames shrunk on the Mac exactly as the server would | 6.14 → 0.41 MB per request; Mac → GPU box 2.40 → 0.30 s (2026-10-18) |
| 3 | **Warm the server** before the first run | the first run after a server start waited 15–17 s for its second answer (likely a first-call compile); later runs never paused over 0.6 s |
| 4 | Server flags | `--actions_to_execute 5 --actions_to_keep 1 --execute_in_n_steps 5 --num_steps 10`: execute 5 of each 30-step chunk, keep 1 as the next chunk's anchor, play them at normal speed (k = 1), 10 denoising steps |
| 5 | Client settings | noise temperature 0.9, inpainting threshold 0.5. The client also sends `cfg_scale 5.0`, which the real model **ignores**: guidance needs the advantage input that the real checkpoint does not have (`pi_modified.py`: `use_cfg = use_advantage_embedding and ...`) |
| 6 | Loop rate | ~11 actions/s on our Mac vs 20 Hz in training: inference is synchronous (0.19 s per 5-action chunk). It still folds |

## 6. LeRobot 0.6.0 and macOS ports of Ilia's runner (`scripts/record_real_dagger_mac.py`)

Moved imports (`lerobot.utils.feature_utils`); vendored `real_robot_config.py`; keyboard via pynput (macOS asks for Input Monitoring; Ctrl-C in the terminal always works); RealSense patch only when a RealSense is configured; `--leader_id`; a missing joint stops the run instead of sending 0.0; the 0.6.0 dataset API (`save_episode()` directly). The Rerun viewer needs the env's `bin/` on PATH (`PATH=~/miniforge3/envs/phi/bin:$PATH python ...`).

## 7. Recording and review

- `calls.jsonl` on every run: our state, the 5-step chunk, and the model's own predictions (garment type, success, completion, checkpoint, time to completion).
- `--max_episode_s` (wall clock) saves the episode when it runs out; the first Ctrl-C saves too, a second stops at once.
- `scripts/stack_streams.py <session>`: one real-speed video with the top camera, the part the model saw, and both wrists.
- In every successful run the model's completion head rose to ~0.95 exactly as the fold landed and the arms parked.

## 8. How to run (one session)

1. GPU box: `scripts/server/b4_serve.sh` in tmux (after checking the GPU is free).
2. Mac: find camera numbers, set the local config, check every arm read-only.
3. `--dry_run` with the garment on the table: the garment guess should be right.
4. Real run:

```bash
PATH="$HOME/miniforge3/envs/phi/bin:$PATH" python scripts/record_real_dagger_mac.py \
  --config configs/real_robot_mac.local.yaml --robot_id phi_bi_follower --leader_id phi_bi_leader \
  --server ws://<gpu-box>:8000 --garment top_short --client_resize --top_crop 0.794 \
  --max_episode_s 75 --start_delay_s 5 --leaders_limp --episodes 1
```

5. `python scripts/stack_streams.py data/lehome_real/recorded/<session>`. Stop the server after the session (it listens on every interface).

## The model, in one table (report sections in brackets)

| | |
|---|---|
| Base | π0.5 extension from Ilia's team's BEHAVIOR-1K 2025 policy: frozen SigLIP image encoder, Gemma-2B prefix, Gemma-300M flow-matching action expert [§1.3, §4.1] |
| Output | 30-step × 12-joint action chunks, deltas from the current state, normalised per timestep; 10 Euler denoising steps [§4.6, §7.1] |
| Inputs | 3 cameras at 224 px, 12 joints (256 bins), a garment-type token instead of a language prompt [§4.1, §4.2] |
| Heads | read image tokens only. Real checkpoint: garment type and completion trained; success, checkpoint, time-to-completion frozen from sim; keypoint and world-model heads removed [§5, §9.5] |
| Sim round | RL: AWR through the data sampler + RECAP-style advantage conditioning, in an asynchronous trainer / rollout / DAgger loop. 1st of 62 at 79.63% [§2, §8] |
| **Real round (this checkpoint)** | **behaviour cloning**, no RL: fine-tuned from a late-but-not-latest sim checkpoint on organiser teleop (60%), Ilia's home teleop + DAgger (30%), augmented sim replays (10%); heavy augmentation; per-source speed factors; DAgger frames weighted by closeness to a human takeover. No advantage, guidance or best-of-N on real. 2nd in the final, 865 points vs 895 [§9] |
| Inference | denoise a chunk, soft inpainting from the previous chunk, garment-type bootstrap (hold the first chunk, latch the predicted type) [§7.2, §7.6, §9.5] |
