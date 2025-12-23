import streamlit as st
import pandas as pd
import subprocess
import numpy as np
import cv2
import mediapipe as mp
import zipfile
from pathlib import Path
import json

FFMPEG_PATH = r"C:\Program Files\WinGet\Links\ffmpeg.exe"
FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg", "ffprobe")

TARGET_RATIO = 9 / 16
OUTPUT_RESOLUTION = (1080, 1920)

CLIPS_DIR = Path("clips")
FINAL_VIDEOS_DIR = Path("final_videos")
CLIPS_DIR.mkdir(exist_ok=True)
FINAL_VIDEOS_DIR.mkdir(exist_ok=True)

# ---------------- TIMESTAMP UTILS (SOURCE-AWARE) ---------------- #

def timestamp_to_seconds(t, source="csv"):
    """
    source="csv"   → MM:SS or MM:SS:00 (40:59:00 = 40 min 59 sec)
    source="video" → HH:MM:SS
    """
    parts = str(t).strip().split(":")

    if source == "csv":
        if len(parts) == 2:      # MM:SS
            m, s = parts
            return int(m) * 60 + int(s)
        if len(parts) == 3:      # MM:SS:00 → ignore last
            m, s, _ = parts
            return int(m) * 60 + int(s)
        raise ValueError(f"Invalid CSV timestamp: {t}")

    if source == "video":
        if len(parts) == 3:      # HH:MM:SS
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        raise ValueError(f"Invalid video timestamp: {t}")

    raise ValueError("source must be 'csv' or 'video'")

def seconds_to_ffmpeg(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------------- VIDEO PROBE (ROBUST) ---------------- #

def probe_video(path):
    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    stream = data["streams"][0]
    w = stream["width"]
    h = stream["height"]

    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    return w, h, fps

# ---------------- ACTIVE SPEAKER ---------------- #

def get_faces(frame, detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    faces = []

    if res.detections:
        h, w, _ = frame.shape
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            cx = int((bb.xmin + bb.width / 2) * w)
            faces.append(cx)

    return faces

def select_active_speaker(prev, curr, frame_w):
    if not curr:
        return None
    if len(curr) == 1:
        return curr[0]
    if prev:
        return curr[np.argmax([min(abs(c - p) for p in prev) for c in curr])]
    return min(curr, key=lambda x: abs(x - frame_w // 2))

# ---------------- STEP 1: CLIP ---------------- #

def clip_video(src, start, end, out):
    ss = timestamp_to_seconds(start, source="csv")
    ee = timestamp_to_seconds(end, source="csv")

    if ee <= ss:
        return False, "End timestamp must be greater than start timestamp"

    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", seconds_to_ffmpeg(ss),
        "-i", str(src),
        "-t", str(ee - ss),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out)
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, r.stderr

    return True, ""

# ---------------- STEP 2+3: PIPE PROCESS ---------------- #

def process_clip_ffmpeg_pipe(input_clip, output_clip, progress_bar):
    w, h, fps = probe_video(input_clip)
    crop_w = int(h * TARGET_RATIO)

    read_cmd = [
        FFMPEG_PATH, "-i", str(input_clip),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]

    write_cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{OUTPUT_RESOLUTION[0]}x{OUTPUT_RESOLUTION[1]}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_clip.with_suffix(".video.mp4")
    ]

    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE)
    writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE)

    frame_size = w * h * 3
    detector = mp.solutions.face_detection.FaceDetection(0, 0.5)

    # ---------------- WINDOW-BASED STABILIZATION STATE ----------------
    last_good_cx = None
    speaker_locked = False

    last_crop_left = max(0, (w - crop_w) // 2)

    DEAD_ZONE = int(w * 0.10)
    SMOOTHING = 0.90
    RIGHT_BIAS_RATIO = 0.80  # ← NEW (bias to speaker side)

    prev_faces = []
    frame_idx = 0
    est_total = int(fps * 60)

    while True:
        raw = reader.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))

        # ---------------- FACE SELECTION ----------------
        faces = get_faces(frame, detector)

        # -------- SPEAKER ACQUISITION + LOCK --------
        if not speaker_locked:
            if faces:
                # Pick one face ONCE (closest to center)
                cx = min(faces, key=lambda x: abs(x - w // 2))
                last_good_cx = cx
                speaker_locked = True
            else:
                # No face yet → bias right
                cx = int(w * RIGHT_BIAS_RATIO)

        else:
            # Speaker already locked
            if not faces:
                # Face lost → bias right instead of center
                cx = int(w * RIGHT_BIAS_RATIO)

            elif len(faces) == 1:
                cx = select_active_speaker(prev_faces, faces, w)
                if cx is None:
                    cx = last_good_cx
                last_good_cx = cx

            else:
                # Multiple faces → HOLD LOCK
                cx = last_good_cx

        prev_faces = faces

        # ---------------- WINDOW-BASED FRAMING ----------------
        desired_left = cx - crop_w // 2
        desired_left = max(0, min(desired_left, w - crop_w))

        # Dead-zone (ignore micro movement)
        if abs(desired_left - last_crop_left) < DEAD_ZONE:
            desired_left = last_crop_left

        # Smooth window motion
        crop_left = int(
            SMOOTHING * last_crop_left +
            (1 - SMOOTHING) * desired_left
        )

        crop_left = max(0, min(crop_left, w - crop_w))
        crop_right = crop_left + crop_w

        last_crop_left = crop_left

        # ---------------- APPLY CROP ----------------
        cropped = frame[:, crop_left:crop_right]
        resized = cv2.resize(
            cropped,
            OUTPUT_RESOLUTION,
            interpolation=cv2.INTER_AREA
        )

        writer.stdin.write(resized.tobytes())
        frame_idx += 1
        progress_bar.progress(min(frame_idx / est_total, 1.0))

    reader.stdout.close()
    writer.stdin.close()
    reader.wait()
    writer.wait()

    # ---------------- AUDIO MUX ----------------
    subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", str(output_clip.with_suffix(".video.mp4")),
        "-i", str(input_clip),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        str(output_clip)
    ], check=True)

    output_clip.with_suffix(".video.mp4").unlink()
    return True, ""




# ---------------- STREAMLIT UI ---------------- #

st.title("🎬 Podcast → Shorts (FFmpeg Pipe Edition)")

video = st.file_uploader("Upload podcast video", type=["mp4"])
csv = st.file_uploader("Upload timestamps CSV", type=["csv"])

if video and csv:
    video_path = Path(video.name)
    video_path.write_bytes(video.getbuffer())

    df = pd.read_csv(csv)
    st.dataframe(df)

    if st.button("🚀 Start Processing"):
        outputs = []

        for i, row in df.iterrows():
            raw = CLIPS_DIR / f"clip_{i:03d}.mp4"
            final = FINAL_VIDEOS_DIR / f"clip_{i:03d}_final.mp4"

            ok, msg = clip_video(
                video_path,
                row["start_timestamp"],
                row["end_timestamp"],
                raw
            )
            if not ok:
                st.error(msg)
                continue

            bar = st.progress(0)
            ok, msg = process_clip_ffmpeg_pipe(raw, final, bar)
            bar.empty()

            if ok:
                outputs.append(final)
            else:
                st.error(msg)

        if outputs:
            zip_path = Path("shorts.zip")
            with zipfile.ZipFile(zip_path, "w") as z:
                for f in outputs:
                    z.write(f, f.name)

            st.download_button(
                "📥 Download Shorts",
                zip_path.read_bytes(),
                "shorts.zip"
            )
