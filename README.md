# 🎬 YouTube Podcast to Shorts Pipeline

This project provides a Streamlit web application to automatically convert long-form podcast videos into vertical (9:16) shorts, perfect for YouTube, TikTok, or Reels.

The application uses AI-powered face tracking to ensure the speaker remains centered in the frame after the video is cropped to a vertical aspect ratio.

## ✨ Features

-   **Automated Clipping:** Clips videos based on timestamps provided in a CSV file.
-   **AI-Powered Face Tracking:** Uses MediaPipe to detect and track the speaker's face.
-   **Dynamic Vertical Cropping:** Intelligently crops the video to a 9:16 aspect ratio, keeping the face centered.
-   **Batch Processing:** Processes multiple timestamps from a CSV file in one go.
-   **Simple Web Interface:** Easy-to-use interface built with Streamlit for uploading files and downloading results.

## 🛠️ Tech Stack

-   **Core Logic:** Python
-   **Web Framework:** Streamlit
-   **Video Processing:** OpenCV & FFmpeg
-   **Face Detection:** MediaPipe
-   **Dependencies:** See `requirements.txt`

## 🚀 Getting Started (Manual Setup)

The environment for this project requires manual setup.

### 1. Install System Dependencies

You must install Python and FFmpeg. Use the following command in the terminal:

```bash
nix-env -iA nixpkgs.python311 nixpkgs.ffmpeg
```

### 2. Install Python Packages

Install the required Python libraries using pip:

```bash
pip install --user -r requirements.txt
```

### 3. Run the Application

Start the Streamlit application with the following command:

```bash
streamlit run app.py
```

The application will then be available in a web preview.

## 📝 How to Use the App

1.  **Upload Video:** Upload your full-length podcast MP4 file.
2.  **Upload CSV:** Upload a CSV file containing `start_timestamp` and `end_timestamp` columns for the clips you want to create.
3.  **Process:** Click the **"Start Processing All Clips"** button.
4.  **Download:** Download the generated shorts individually or as a single `.zip` file.
