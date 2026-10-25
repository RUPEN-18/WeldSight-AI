
# WeldSight AI

AI-assisted visual weld inspection using YOLO26s.

## Features
- Weld defect detection from images
- Welding video analysis at approximately 8 FPS
- Confidence-based detection filtering
- LOW, MEDIUM, and HIGH visual severity estimates
- Automatic PDF inspection reports
- Windows desktop application
- Standalone Windows executable built using PyInstaller

## Detected Categories
1. Bad Welding
2. Crack
3. Excess Reinforcement
4. Good Welding
5. Porosity
6. Spatters

## Model and Training
- Model: YOLO26s
- Training epochs: 20
- Image size: 640 × 640
- Batch size: 4
- Training device: CPU
- Dataset split: 1,109 training, 191 validation,
  and 89 test images

## Technology Stack
Python, Ultralytics, PyTorch, OpenCV, PySide6,
ReportLab, and PyInstaller.

## Limitations
This is an AI-assisted visual inspection prototype.
Results depend on image quality, lighting, and defect type.
Visual severity is an estimate, not an engineering acceptance
rating. Qualified inspection and applicable NDT methods
are still required for final weld acceptance.

## Model Weights
Trained model weights are not included in this repository.
An authorized model file must be placed at the path expected
by the application before running it.
