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

- Player detection using YOLOv5 and custom model fine-tuned on yolov11
- Attempts at consistent player ID assignment across frames
- Partial handling of occlusions and re-identification
- Customizable confidence and IoU thresholds
- Support for video file input
- Separate pipelines for testing different models (best.pt vs YOLOv5)

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
git clone https://github.com/Anushka197/visionary-eyes/
cd visionary-eyes
```

2. Create and activate a virtual environment:

```bash
python -m venv cv_env
cv_env\Scripts\activate       # Windows
          OR
source cv_env/bin/activate    # macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your model weights:
   - yolov5 model in `task2/`
   - Custom model (`best.pt`) in `resources/`

---

## Project Structure

```
visionary-eyes/
│
├── task2/                        # YOLOv5
│   ├── yolov5s.pt                # Trained model
│   ├── config.json               # Configuration for this task
│   ├── detect_player.py          # Player detection logic
│   ├── identify.py               # Tracking + ID assignment
│   ├── color_detection.py        # Color clustering (optional)
│   ├── output/                   # Output videos/images
│   └── yolov5/                   # YOLOv5 repo (submodule)
│
├── original_task2/               # best.py
│   ├── config.json               # Separate config file
│   ├── detection.py              # detection
│   ├── identification.py         # Basic ID assignment
│   ├── test_model.py             # Model features check
│   ├── model_inspection.py       # Analyze model (layers/params)
│   ├── model_info.txt            # output from model_inspection.py
│   ├── output/                   # Output for this task
│   ├── runs/                     # experiment logs
│   └── detection_conf_iou.py     # Threshold tuning
│
├── resources/                    # Common resources, videos and best.pt
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

### Run best.pt-based detection (`original_task2`):

```bash
cd original_task2
python custom_id.py
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
