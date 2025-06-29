# Visionary Eyes

Perceiving the world as it is — and as it could be.

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Project Structure](#project-structure)  
- [Usage](#usage)  
- [Output](#output)  
- [License](#license)

---

## Overview

Visionary Eyes is built to improve the reliability of player tracking in sports analytics. It attempts to tackle the common problem of ID switching by combining object detection with basic tracking and identity association techniques.

This is an experimental project — not production-ready. The aim is to prototype different strategies and understand their limitations in real-world scenarios.

---

## Features

- Player detection using YOLOv5 and custom model pretrained on yolov11
- Attempts at consistent player ID assignment across frames
- Partial handling of occlusions and re-identification
- Customizable confidence and IoU thresholds
- Support for video file input
- Separate pipelines for testing different models (custom vs YOLOv8)

---

## Requirements

Dependencies include:

- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- Torch, torchvision
- NumPy, Pandas

Optional:

- scikit-learn (for color clustering)
- OCR tools (Tesseract, EasyOCR)
- Deep SORT (for improved tracking)

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/visionary-eyes.git
cd visionary-eyes
```

2. Create and activate a virtual environment:

```bash
python -m venv cv_env
cv_env\Scripts\activate       # Windows
# or
source cv_env/bin/activate    # macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your model weights:
   - yolov8 model in `task2/`
   - Custom model (``best.pt`) in `resources/`

---

## Project Structure

```
visionary-eyes/
│
├── task2/                        # YOLOv5 pipeline (with Deep-Person-ReID optional)
│   ├── yolov5s.pt                # Trained model
│   ├── config.json               # Configuration for this task
│   ├── detect_player.py          # Player detection logic
│   ├── identify.py               # Tracking + ID assignment
│   ├── color_detection.py        # Color clustering (optional)
│   ├── debug_crops/              # For visual debugging
│   ├── output/                   # Output videos/images
│   └── yolov5/                   # YOLOv5 repo (submodule or clone)
│
├── original_task2/              # YOLOv8 pipeline using Ultralytics
│   ├── config.json               # Separate config file
│   ├── detection.py              # YOLOv8 detection
│   ├── identification.py        # Basic ID assignment
│   ├── test_model.py            # Model performance check
│   ├── model_inspection.py      # Analyze layers/params
│   ├── model_info.txt           # Notes/stats
│   ├── output/                   # Output for this task
│   ├── runs/                     # YOLOv8 experiment logs
│   └── detection_conf_iou.py    # Threshold tuning
│
├── resources/                   # Common resources, videos and best.pt
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Usage

### Run YOLOv5-based detection (`task2`):

```bash
cd task2
python identify.py
```

Modify `task2/config.json` to change video path, confidence, output directory, etc.

### Run YOLOv8-based detection (`original_task2`):

```bash
cd original_task2
python identification.py
```

Edit `original_task2/config.json` to set detection parameters and input sources.

---

## Output

Processed videos with bounding boxes and assigned IDs will be saved to the `output/` directory inside each task folder (`task2/output` or `original_task2/output`).

Each frame contains:
- Bounding boxes
- Player ID

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
