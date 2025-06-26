import torch
import cv2
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
os.makedirs("debug_crops", exist_ok=True)

# === Load config ===
with open('config.json', 'r') as f:
    config = json.load(f)

# === Extract values ===
video_path = config["video"]["input_path"]
output_path = config["video"]["output_path"]
display_output = config["video"]["display_output"]
save_results = config["video"]["save_results"]

conf_thresh = config["detection"]["model"]["confidence_threshold"]
iou_thresh = config["detection"]["model"]["iou_threshold"]
model_name = config["detection"]["model"]["name"]
classes_to_detect = config["detection"]["classes_to_detect"]

draw_color = tuple(config["detection"]["visual"]["draw_box_color"])
font_scale = config["detection"]["visual"]["font_scale"]
font_thickness = config["detection"]["visual"]["font_thickness"]
label_prefix = config["detection"]["visual"]["label_prefix"]

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
model.conf = conf_thresh
model.iou = iou_thresh

# === Initialize Deep SORT Tracker ===
tracker = DeepSort(
    max_age=3,                 # Keep IDs alive longer
    n_init=1,
    max_cosine_distance=0.5,   # Feature similarity threshold
    nn_budget=100,             # Store recent embeddings
    override_track_class=None,
    embedder="mobilenet",      # CPU-friendly
    half=True,
    bgr=True,
)

# === Open video ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if config["video"]["frame_rate"] is None:
    config["video"]["frame_rate"] = fps
if config["video"]["resolution"]["width"] is None:
    config["video"]["resolution"]["width"] = width
    config["video"]["resolution"]["height"] = height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Frame processing ===
def process_frame(frame):
    results = model(frame)
    detections = results.xyxy[0]

    track_inputs = []

    for *box, conf, cls in detections:
        class_id = int(cls.item())
        class_name = model.names[class_id]

        if class_name in classes_to_detect:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            # Skip invalid boxes
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue

            cropped = frame[y1:y2, x1:x2]

            # Ensure crop is non-empty and big enough for ReID
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue

            # Resize for standard ReID embedding size
            cropped_resized = cv2.resize(cropped, (128, 256))
            # cv2.imwrite(f"debug_crops/frame{frame_id}_x{x1}_{y1}_{x2}_{y2}.jpg", cropped_resized)

            track_inputs.append(([x1, y1, w, h], conf.item(), cropped_resized))

    if len(track_inputs) == 0:
        return frame

    tracks = tracker.update_tracks(track_inputs, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        label = f"{label_prefix} {track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, draw_color, font_thickness)

    return frame

frame_id = 0
# === Main processing loop (no threads for consistency) ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed = process_frame(frame)
    out.write(processed)

    if display_output:
        cv2.imshow("Tracking", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_id += 1


cap.release()
out.release()
cv2.destroyAllWindows()