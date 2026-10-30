"""One video of a recorded episode, all cameras at real speed: top (with the crop box), what the model saw, both wrists.

  python scripts/stack_streams.py data/lehome_real/recorded/<session> [--top_crop 0.794]

Writes <session>/all_streams.mp4 and opens it. Layout: top row = the top camera as recorded, the red box = the part
sent to the model (--top_crop) | that part alone; bottom row = left wrist | right wrist. The recording stamps frames
at --fps (20) but the loop runs slower, so the real duration is taken from calls.jsonl (the model calls' times).
Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def real_duration_s(session: Path) -> float | None:
    f = session / "calls.jsonl"
    if not f.exists():
        return None
    t = [json.loads(line)["t"] for line in f.read_text().splitlines() if line.strip()]
    return t[-1] - t[0] if len(t) > 1 else None


def video_duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--top_crop", type=float, default=0.794)
    ap.add_argument("--no_open", action="store_true")
    a = ap.parse_args()
    v = a.session / "videos"
    top = v / "observation.images.right_front/chunk-000/file-000.mp4"
    left = v / "observation.images.left_wrist/chunk-000/file-000.mp4"
    right = v / "observation.images.right_wrist/chunk-000/file-000.mp4"
    for p in (top, left, right):
        if not p.exists():
            raise SystemExit(f"missing {p}: was the episode saved?")
    real, rec = real_duration_s(a.session), video_duration_s(top)
    k = real / rec if real else 1.0  # slow the recording down to wall-clock speed
    c = a.top_crop
    # crop box in the recorded frame: centre fraction c, as the runner's PolicyClient._top
    box = (f"drawbox=x=(iw-iw*{c})/2:y=(ih-ih*{c})/2:w=iw*{c}:h=ih*{c}:color=red:t=6")
    crop = f"crop=iw*{c}:ih*{c}:(iw-iw*{c})/2:(ih-ih*{c})/2"
    filt = (f"[0:v]split=2[t0][t1];"
            f"[t0]{box},scale=640:360,setpts=PTS*{k}[full];"
            f"[t1]{crop},scale=640:360,setpts=PTS*{k}[model];"
            f"[1:v]scale=480:360,pad=640:360:80:0,setpts=PTS*{k}[l];"
            f"[2:v]scale=480:360,pad=640:360:80:0,setpts=PTS*{k}[r];"
            f"[full][model]hstack[a];[l][r]hstack[b];[a][b]vstack,format=yuv420p[o]")
    out = a.session / "all_streams.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(top), "-i", str(left), "-i", str(right),
                    "-filter_complex", filt, "-map", "[o]", "-c:v", "libx264", "-crf", "22", "-preset", "veryfast",
                    str(out)], check=True)
    print(f"{out}  ({video_duration_s(out):.1f} s; recording {rec:.1f} s slowed x{k:.2f} to real time)")
    if not a.no_open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
