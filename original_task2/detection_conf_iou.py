from ultralytics import YOLO
import cv2
import time
import os
import csv

# === Load the model ===
model = YOLO("..\\resources\\best.pt")

# === Open video ===
video_path = "..\\resources\\15sec_input_720p.mp4"
output_video_path = ".\\output\\detection.mp4"
output_csv_path = ".\\output\\detection_results.csv"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Prepare video writer ===
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# === Prepare CSV writer ===
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
csv_columns = ["frame", "id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
csv_data = []

# === Start processing ===
start = time.time()
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(
        source=frame,
        conf=0.8,  # confidence threshold
        iou=0.5,   # NMS IoU threshold
        verbose=False
    )

    # Draw results + collect for CSV
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls] if cls in model.names else f"class_{cls}"
            iou_val = float(box.iou[0]) if hasattr(box, 'iou') else 0.5

            # Draw box
            label = f"conf={conf:.2f} | iou={iou_val:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save to CSV (id = -1 because it's detection only)
            csv_data.append([frame_id, -1, class_name, conf, x1, y1, x2, y2])

    # Output frame
    out.write(frame)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# === Save CSV ===
with open(output_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_columns)
    writer.writerows(csv_data)

# === Clean up ===
end = time.time()
total = (end - start) / 60
print(f"Total Time: {total:.2f} mins")
print(f"CSV saved to: {output_csv_path}")

cap.release()
out.release()
cv2.destroyAllWindows()
