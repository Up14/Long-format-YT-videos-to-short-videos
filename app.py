import streamlit as st
import pandas as pd
import subprocess
import os
import cv2
import mediapipe as mp
import numpy as np
import zipfile
from pathlib import Path
FFMPEG_PATH = r"C:\Program Files\WinGet\Links\ffmpeg.exe"
# --- Constants and Configuration ---
TARGET_RATIO = 9 / 16
OUTPUT_RESOLUTION = (1080, 1920)
CLIPS_DIR = Path("clips")
FINAL_VIDEOS_DIR = Path("final_videos")

# --- Timestamp Normalization ---
def normalize_timestamp(t):
    """Converts MM:SS or other formats to HH:MM:SS for FFmpeg."""
    t = str(t).strip()
    parts = t.split(":")
    try:
        if len(parts) == 2: return f"00:{parts[0]}:{parts[1]}"
        if len(parts) == 3:
            if int(parts[0]) > 23: return f"00:{parts[0]}:{parts[1]}"
            return t
        raise ValueError(f"Invalid timestamp format: {t}")
    except (ValueError, IndexError): raise ValueError(f"Invalid timestamp format: {t}")

# --- Face Detection Logic (Used only for analysis) ---
def get_face_center(frame, face_detector):
    """Detects the primary face and returns its center x-coordinate."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        return int((bbox.xmin + bbox.width / 2) * w)
    return None

# --- CORRECT ARCHITECTURE: Core Video Processing Functions ---

def clip_video(input_video_path, start_time, end_time, output_path):
    """Step 1: Clips a video using FFmpeg, preserving audio."""
    cmd = [
        FFMPEG_PATH, "-i", str(input_video_path), "-ss", str(start_time), "-to", str(end_time),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-c:a", "copy", "-y", str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as e: return False, e.stderr

def analyze_clip_for_crop(clip_path, progress_bar):
    """Step 2: Analyzes the clip with OpenCV to find the best stable crop, but does NOT write video."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened(): return None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_centers = []

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            # Sample every 10 frames for efficiency and stability
            if frame_num % 10 == 0:
                center = get_face_center(frame, detector)
                if center: face_centers.append(center)
            progress_bar.progress(frame_num / total_frames)

    cap.release()
    
    if not face_centers:
        # If no faces found, default to the center of the frame.
        return frame_width // 2, frame_height
    
    # Return the average face position for a stable crop.
    return int(np.mean(face_centers)), frame_height

def apply_ffmpeg_crop(input_clip_path, output_path, crop_center_x, frame_height):
    """Step 3: Uses FFmpeg to apply the calculated crop, scale, and copy audio for perfect sync."""
    crop_width = int(frame_height * TARGET_RATIO)
    x_coord = max(0, crop_center_x - crop_width // 2)

    # Ensure the crop coordinates are integers
    crop_width = int(crop_width)
    x_coord = int(x_coord)

    video_filter = f"crop={crop_width}:{frame_height}:{x_coord}:0,scale={OUTPUT_RESOLUTION[0]}:{OUTPUT_RESOLUTION[1]}"

    cmd = [
        FFMPEG_PATH, "-y", "-i", str(input_clip_path),
        "-vf", video_filter,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "copy", # CRITICAL: Copy audio stream without re-encoding
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as e: return False, e.stderr

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🎬 YouTube Podcast to Shorts Pipeline")
CLIPS_DIR.mkdir(exist_ok=True); FINAL_VIDEOS_DIR.mkdir(exist_ok=True)

st.header("1. Upload Your Files")
uploaded_video = st.file_uploader("Upload Long-Form Podcast Video (MP4)", type=["mp4"])
uploaded_csv = st.file_uploader("Upload Timestamps CSV", type=["csv"])

if uploaded_video and uploaded_csv:
    video_path = Path(f"temp_{uploaded_video.name}")
    with open(video_path, "wb") as f: f.write(uploaded_video.getbuffer())

    try:
        df = pd.read_csv(uploaded_csv)
        st.header("2. Review Timestamps"); st.dataframe(df)

        if st.button("🚀 Start Processing All Clips"):
            st.header("3. Processing and Downloads")
            final_video_paths = []

            for index, row in df.iterrows():
                clip_identifier = f"clip_{index:03d}"
                raw_clip_path = CLIPS_DIR / f"{clip_identifier}.mp4"
                final_video_path = FINAL_VIDEOS_DIR / f"{clip_identifier}_final.mp4"
                
                try:
                    start_time = normalize_timestamp(row['start_timestamp'])
                    end_time = normalize_timestamp(row['end_timestamp'])
                except ValueError as e: st.error(f"Skipping row {index}: {e}"); continue

                st.subheader(f"Processing Clip {index + 1}/{len(df)}: {start_time} - {end_time}")
                
                # STEP 1: Clip with perfect audio
                with st.spinner(f"Step 1/2: Clipping segment..."):
                    success, msg = clip_video(video_path, start_time, end_time, raw_clip_path)
                    if not success: st.error(f"Failed Step 1 (Clip):"); st.code(msg); continue
                    st.success("Step 1/2: Clipping successful.")

                # STEP 2 & 3: Analyze and then Crop with FFmpeg
                if raw_clip_path.exists():
                    progress_bar = st.progress(0)
                    with st.spinner(f"Step 2/2: Analyzing for stable crop & finalizing video..."):
                        # Analyze to get crop coordinates
                        crop_center, frame_h = analyze_clip_for_crop(raw_clip_path, progress_bar)
                        if crop_center is None:
                            st.warning(f"Could not analyze clip {index}. Skipping.")
                            continue
                        
                        # Apply crop using FFmpeg for perfect audio sync
                        success, msg = apply_ffmpeg_crop(raw_clip_path, final_video_path, crop_center, frame_h)
                        if success:
                            st.success("Step 2/2: Analysis and finalization complete!")
                            final_video_paths.append(final_video_path)
                        else:
                            st.error(f"Failed Step 2 (Crop/Finalize):"); st.code(msg)
                    progress_bar.empty() # Clean up progress bar
                else: st.error(f"Raw clip for row {index} not found after clipping.")
                st.markdown("---")

            # FINAL DOWNLOAD SECTION
            if final_video_paths:
                st.header("4. Download All Processed Clips")
                zip_path = Path("processed_shorts.zip")
                with st.spinner("Zipping all videos for download..."):
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fp in final_video_paths:
                            if fp.exists(): zf.write(fp, fp.name)
                with open(zip_path, "rb") as f:
                    st.download_button("📥 Download All Clips as .zip", f.read(), "youtube_shorts_batch.zip", "application/zip")
    finally:
        if video_path.exists(): video_path.unlink()

st.sidebar.title("About")
st.sidebar.info("This app automates creating YouTube Shorts from long-form videos.")
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """1. Upload your video and CSV.\n2. Click 'Start Processing'.\n3. A ZIP download button will appear."""
)