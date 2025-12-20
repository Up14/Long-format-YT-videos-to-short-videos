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
    """
    Step 2: Analyzes the clip using "Segmented Reframing" to create a smooth, dynamic camera path.
    Returns a list of integer x-coordinates, one for each frame.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, None

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- Configuration for "Segmented Reframing" ---
    SEGMENT_DURATION_SECONDS = 3
    FRAMES_PER_SEGMENT = int(SEGMENT_DURATION_SECONDS * fps)
    MOVE_THRESHOLD = frame_width * 0.08  # Move if target is > 8% of frame width away
    TRANSITION_DURATION_FRAMES = int(0.5 * fps) # 0.5-second smooth transition
    FACE_SAMPLE_RATE = 5 # Sample faces every 5 frames within a segment

    # --- Analysis ---
    segment_medians = []
    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        for start_frame in range(0, total_frames, FRAMES_PER_SEGMENT):
            end_frame = min(start_frame + FRAMES_PER_SEGMENT, total_frames)
            segment_centers = []

            # Set video capture to the start of the segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample faces at a specific rate for efficiency
                if frame_num % FACE_SAMPLE_RATE == 0:
                    center = get_face_center(frame, detector)
                    if center:
                        segment_centers.append(center)
                
                # Update progress bar based on overall progress
                progress_bar.progress(frame_num / total_frames, text="Step 2/3: Analyzing clip...")


            if segment_centers:
                # Use median for robustness against outliers
                median_center = int(np.median(segment_centers))
                segment_medians.append(median_center)
            else:
                # If no face found, use the previous segment's median or default to center
                last_median = segment_medians[-1] if segment_medians else frame_width // 2
                segment_medians.append(last_median)

    cap.release()

    if not segment_medians:
        # If no faces were ever found, create a static crop centered in the frame
        return [frame_width // 2] * total_frames, frame_height

    # --- Generate Smoothed Camera Path ---
    camera_path = []
    current_center = segment_medians[0] # Start with the first segment's median
    camera_path.extend([current_center] * FRAMES_PER_SEGMENT) # Fill the first segment

    for i in range(1, len(segment_medians)):
        target_center = segment_medians[i]
        
        # Decide if camera should move
        if abs(target_center - current_center) > MOVE_THRESHOLD:
            # Smooth transition
            transition_points = np.linspace(current_center, target_center, TRANSITION_DURATION_FRAMES)
            
            # Fill frames before transition
            frames_before_transition = FRAMES_PER_SEGMENT - TRANSITION_DURATION_FRAMES
            if frames_before_transition > 0:
                camera_path.extend([current_center] * frames_before_transition)

            # Add the transition frames
            camera_path.extend(transition_points.astype(int))
            current_center = target_center # Update camera position
        else:
            # No movement, hold the current position
            camera_path.extend([current_center] * FRAMES_PER_SEGMENT)

    # Ensure camera_path has the correct number of frames, trimming or padding if necessary
    camera_path = camera_path[:total_frames]
    while len(camera_path) < total_frames:
        camera_path.append(camera_path[-1])
        
    return camera_path, frame_height

def apply_dynamic_crop_and_save(input_clip_path, output_path, camera_path, frame_height, progress_bar):
    """
    Step 3: Reads the clip frame-by-frame, applies the dynamic crop from the camera_path,
    and saves the result. Finally, it merges the audio from the original clip.
    """
    cap = cv2.VideoCapture(str(input_clip_path))
    if not cap.isOpened():
        return False, "Could not open input clip for cropping."

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Path for the temporary video file (no audio)
    temp_video_path = input_clip_path.with_suffix('.temp.mp4')
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'avc1'
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, OUTPUT_RESOLUTION)

    if not out.isOpened():
        cap.release()
        return False, "Could not open VideoWriter for temporary file."

    crop_width = int(frame_height * TARGET_RATIO)

    # Process each frame
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Get the target center for this frame from the pre-computed camera path
        center_x = camera_path[frame_num]
        
        # Calculate crop coordinates
        x1 = max(0, center_x - crop_width // 2)
        x2 = min(frame_width, x1 + crop_width)
        
        # Adjust x1 if crop goes out of bounds
        if x2 - x1 < crop_width:
            x1 = x2 - crop_width

        # Perform the crop
        cropped_frame = frame[:, int(x1):int(x2)]

        # Resize to the final output resolution
        resized_frame = cv2.resize(cropped_frame, OUTPUT_RESOLUTION, interpolation=cv2.INTER_AREA)

        out.write(resized_frame)
        progress_bar.progress(frame_num / total_frames, text="Step 3/3: Applying crop and finalizing video...")

    # Release resources
    cap.release()
    out.release()
    
    # --- Step 4: Merge audio from original clip using FFmpeg ---
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(temp_video_path),  # Video from OpenCV
        "-i", str(input_clip_path),  # Audio from original clip
        "-c:v", "copy",              # Copy the already encoded video stream
        "-c:a", "aac",               # Re-encode audio to a standard format (or use "copy")
        "-map", "0:v:0",             # Map video from the first input
        "-map", "1:a:0",             # Map audio from the second input
        "-shortest",                 # Finish encoding when the shortest stream ends
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Clean up the temporary video file
        if temp_video_path.exists():
            temp_video_path.unlink()
        return True, ""
    except subprocess.CalledProcessError as e:
        # Clean up on failure as well
        if temp_video_path.exists():
            temp_video_path.unlink()
        return False, f"FFmpeg audio merge failed: {e.stderr}"

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
                with st.spinner(f"Step 1/3: Clipping segment..."):
                    success, msg = clip_video(video_path, start_time, end_time, raw_clip_path)
                    if not success: st.error(f"Failed Step 1 (Clip):"); st.code(msg); continue
                    st.success("Step 1/3: Clipping successful.")

                # STEP 2 & 3: Analyze and then Crop
                if raw_clip_path.exists():
                    # Create a single progress bar for the next two steps
                    progress_bar = st.progress(0, text="Starting analysis...")
                    
                    # STEP 2: Analyze for dynamic crop
                    camera_path, frame_h = analyze_clip_for_crop(raw_clip_path, progress_bar)
                    if camera_path is None:
                        st.warning(f"Could not analyze clip {index}. Skipping.")
                        progress_bar.empty()
                        continue
                    st.success("Step 2/3: Clip analysis complete.")
                    
                    # STEP 3: Apply dynamic crop and save
                    success, msg = apply_dynamic_crop_and_save(raw_clip_path, final_video_path, camera_path, frame_h, progress_bar)
                    if success:
                        st.success("Step 3/3: Dynamic crop applied and video finalized!")
                        final_video_paths.append(final_video_path)
                    else:
                        st.error(f"Failed Step 3 (Crop/Finalize):"); st.code(msg)
                    
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
