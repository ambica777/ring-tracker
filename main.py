import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ========= CONFIG =========
VIDEO_PATH = "input/anna_sample1.mp4"  # Your test video
OUTPUT_DIR = "output"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "output_multi_fashion_final1.mp4")
MODEL_PATH = "runs/detect/train18/weights/best.pt"  # Path to your trained model
CONF_THRESHOLD = 0.25

# ========= SETUP =========
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

# ========= READ VIDEO =========
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Failed to open input video")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# ========= PROCESS FRAMES =========
for i in tqdm(range(frame_count), desc="Detecting Fashion Items"):
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    detections = results[0]

    for box in detections.boxes:
        conf = box.conf.item()
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add frame number
    cv2.putText(frame, f"Frame {i+1}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Done. Output saved to: {OUTPUT_VIDEO}")
