from ultralytics import YOLO
import cv2
import os
import numpy as np

# === Setup ===
video_path = "..\\resources\\15sec_input_720p.mp4"
model_path = "..\\resources\\best.pt"
output_path = ".\\output\\manual_id_output.avi"
os.makedirs("./output", exist_ok=True)

# === Load model ===
model = YOLO(model_path)

# === Start tracking stream ===
results = model.track(
    source=video_path,
    persist=True,
    conf=0.8,
    iou=0.5,
    stream=True
)

# === Output writer ===
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))

# === Manual ID Map ===
id_map = {}       # maps YOLO track ID → custom ID
next_manual_id = 1

# === Processing frames ===
for result in results:
    frame = result.orig_img

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            conf = confs[i]
            raw_track_id = int(track_ids[i]) if track_ids is not None else None

            # === Manual ID Assignment ===
            if raw_track_id not in id_map:
                id_map[raw_track_id] = next_manual_id
                next_manual_id += 1

            manual_id = id_map[raw_track_id]
            label = f"Player {manual_id}"

            # === Draw Box & Label ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === Save the frame ===
    out.write(frame)
    cv2.imshow("Manual ID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
