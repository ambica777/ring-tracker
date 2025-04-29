# Thuli Studios - Ring, Earring, and Dress Detection Pipeline

## Problem Statement
Build a real-time fashion accessory tracking pipeline focusing on rings on fingers, earrings on ears, and dresses on body using computer vision and deep learning models.

## Solution Summary
We developed a complete YOLOv8-based object detection pipeline trained on a custom merged dataset of rings, earrings, and dresses. It performs inference on videos with real-world hand movements.

## Project Structure
- datasets/multi-fashion-dataset/  → Final merged dataset
- runs/detect/train15/              → YOLOv8 trained model
- scripts/
  - main.py                         → Final inference script
  - merge.py                        → Dataset merger script
  - rewrite.py                      → Label remapping script
- output/                           → Output videos
- README.md                         → Project overview
- Design_Document.docx              → Full technical breakdown

## Setup Instructions
```bash
conda activate ring-tracker
pip install ultralytics opencv-python tqdm
python scripts/main.py
