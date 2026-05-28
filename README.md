# Human Squat Quality Detection using Image and Video Processing

An AI-powered biomechanical squat assessment system that evaluates squat form from videos using computer vision, pose estimation, temporal feature extraction, and deep learning.

## Overview

This project combines image processing, biomechanics, and deep learning to analyze squat quality in real time.

The system extracts 3D human pose landmarks using MediaPipe, processes temporal movement patterns, and evaluates squat performance using a hybrid pipeline consisting of biomechanical heuristics and an Attention-based BiLSTM model.

## Features

* 3D pose estimation using MediaPipe
* Real-time squat quality assessment
* Automatic repetition detection
* Temporal biomechanical feature extraction
* Attention-based BiLSTM architecture
* Hybrid heuristic + ML evaluation pipeline
* Visual feedback overlays
* Posture, depth, and stability analysis

## Tech Stack

* Python
* OpenCV
* MediaPipe
* PyTorch
* NumPy
* SciPy

## Project Pipeline

1. Video Input
2. Pose Landmark Extraction
3. 3D Pose Normalization
4. Temporal Feature Engineering
5. Motion Phase Segmentation
6. Attention-based Sequence Learning
7. Hybrid Squat Evaluation

## Installation

```bash
pip install -r requirements.txt
```

## Run Inference

```bash
python video_inference.py --video input_video.mp4
```

## Output

The system generates:

* Squat quality score
* Biomechanical diagnostics
* Stability analysis
* Annotated output video

## Future Improvements

* Real-time webcam inference
* Mobile deployment
* Additional exercise detection
* Injury-risk estimation

## Author

KARSHIN
SRIHARI
SANKAR
