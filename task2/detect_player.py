import torch
import cv2
import json
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Load and parse the config ===
with open('config.json', 'r') as f:
    config = json.load(f)

# === Extract config values ===
video_config = config["video"]
detection_config = config["detection"]
model_config = detection_config["model"]
visual_config = detection_config["visual"]

video_path = "..\\resources\\15sec_input_720p.mp4"
output_path = ".\\output\\detection.mp4"
csv_output_path = ".\\output\\detection_results.csv"
display_output = video_config["display_output"]
save_results = video_config["save_results"]

conf_thresh = model_config["confidence_threshold"]
iou_thresh = model_config["iou_threshold"]
model_name = model_config["name"]

classes_to_detect = detection_config["classes_to_detect"]
draw_color = tuple(visual_config["draw_box_color"])
font_scale = visual_config["font_scale"]
font_thickness = visual_config["font_thickness"]
label_prefix = visual_config["label_prefix"]

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
model.conf = conf_thresh
model.iou = iou_thresh

# === Open video and get metadata ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Update config resolution if needed
if video_config["frame_rate"] is None:
    video_config["frame_rate"] = fps
if video_config["resolution"]["width"] is None:
    video_config["resolution"]["width"] = width
    video_config["resolution"]["height"] = height

# === Prepare video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f"FPS: {fps}, Width: {width}, Height: {height}, Opened: {cap.isOpened()}")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Prepare CSV logging ===
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
csv_data = []
csv_columns = ["frame", "class_name", "confidence", "x1", "y1", "x2", "y2"]

# === Thread-safe detection function ===
def process_frame(frame, frame_num):
    results = model(frame)
    detections = results.xyxy[0]
    frame_data = []

    for *box, conf, cls in detections:
        class_id = int(cls.item())
        class_name = model.names[class_id]

        if class_name in classes_to_detect:
            x1, y1, x2, y2 = map(int, box)
            label = f"{label_prefix} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, draw_color, font_thickness)

            # Store for CSV
            frame_data.append([frame_num, class_name, float(conf), x1, y1, x2, y2])

    return frame, frame_data

# === Thread pool executor ===
executor = ThreadPoolExecutor(max_workers=4)
futures = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Submit frame for threaded processing
    future = executor.submit(process_frame, frame.copy(), frame_id)
    futures.append(future)
    frame_id += 1

    # Process one at a time to avoid memory pile-up
    if len(futures) >= 4:
        done, futures = futures[0:1], futures[1:]
        processed_frame, frame_data = done[0].result()
        csv_data.extend(frame_data)
        out.write(processed_frame)
        if display_output:
            cv2.imshow("Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Finalize any remaining frames
for future in futures:
    processed_frame, frame_data = future.result()
    csv_data.extend(frame_data)
    out.write(processed_frame)
    if display_output:
        cv2.imshow("Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# === Write CSV file ===
with open(csv_output_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_columns)
    writer.writerows(csv_data)

# === Clean up ===
cap.release()
out.release()
cv2.destroyAllWindows()
executor.shutdown()
print(f"\nDetection complete. Results saved to: {csv_output_path}")
