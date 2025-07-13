# 🎯 Player Re-Identification in Sports Footage

This project tackles the challenge of **cross-camera player re-identification** in sports by analyzing two synchronized video feeds — a **broadcast view** and a **tacticam view**. It uses object detection and histogram-based appearance matching to identify and match players across different camera angles.

---

## 📽️ Project Overview

- Detect players in both broadcast and tacticam videos using a YOLOv1-based detector.
- Extract HSV color histograms for each detected player.
- Match players from tacticam to broadcast view based on histogram similarity.
- Print out match results for each frame and visualize HSV features for analysis.

This system is useful for sports analytics, player tracking, and automated video annotation.

---

## 🧠 Methodology

### 1. Object Detection
- Utilizes **YOLOv1 (Ultralytics)** model for detecting players.
- Player bounding boxes are extracted per frame.

### 2. Frame Processing
- Only every **5th frame** is analyzed to improve performance.
- Processing is limited to **100 frames** per video to manage computational cost.

### 3. Feature Extraction
- Computes **HSV color histograms** from detected bounding boxes.
- Normalized histograms are used for appearance-based matching.

### 4. Player Matching
- For each frame, compares tacticam detections to broadcast ones using OpenCV’s `compareHist` with **correlation metric**.
- Highest scoring pair is selected as the best match.

### 5. Result Output
- Outputs mapping of each tacticam player to a broadcast player with a similarity score.
- Handles frames where no detections are found.
- Visualizes HSV histograms for initial frames to inspect feature quality.

---

## ✅ Features

- 🧍 Accurate **player detection** using YOLO.
- 🧪 Lightweight **appearance-based matching** using HSV histograms.
- 🕵️‍♂️ Detects and matches players across **camera views**.
- 📉 Displays **similarity scores** and **matching results** per frame.
- 📊 Optionally displays **HSV histograms** for detected players.

---

## 🛠️ Installation

**Requirements:**
- Python 3.7+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Matplotlib

**Install dependencies:**

```bash
pip install -r requirements.txt
