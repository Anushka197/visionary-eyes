import torch
import cv2
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Create debug directory ===
os.makedirs("debug_crops", exist_ok=True)

# === Load config ===
with open('config.json', 'r') as f:
    config = json.load(f)

# === Extract values ===
video_path = "..\\resources\\15sec_input_720p.mp4"
output_path = ".\\output\\clrIdentify.mp4"
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
    max_age=10,
    n_init=1,
    max_cosine_distance=0.5,
    nn_budget=100,
    override_track_class=None,
    embedder="mobilenet",
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

# === Track ID remapping ===
id_map = {}
next_id = 0

# === Function: Get dominant jersey color ===
def get_dominant_color(image, k=1):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((-1, 3))
        clt = KMeans(n_clusters=k, n_init=10)
        clt.fit(image)
        return tuple(map(int, clt.cluster_centers_[0]))  # RGB
    except:
        return (127, 127, 127)

# === Frame processing ===
def process_frame(frame):
    global id_map, next_id

    results = model(frame)
    detections = results.xyxy[0]
    track_inputs = []

    for *box, conf, cls in detections:
        class_id = int(cls.item())
        class_name = model.names[class_id]

        if class_name in classes_to_detect:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue

            # === Resize full body crop ===
            cropped_resized = cv2.resize(cropped, (128, 256))

            # === Extract jersey region (upper half) ===
            jersey_roi = cropped[0:int(cropped.shape[0] * 0.5), :]
            jersey_color = get_dominant_color(jersey_roi)

            # === Create a vertical color bar (BGR) and attach to the crop ===
            color_patch = np.full((256, 32, 3), jersey_color[::-1], dtype=np.uint8)
            combined_image = np.hstack([cropped_resized, cv2.resize(color_patch, (32, 256))])

            track_inputs.append(([x1, y1, w, h], conf.item(), combined_image))

    if len(track_inputs) == 0:
        return frame

    tracks = tracker.update_tracks(track_inputs, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        raw_id = track.track_id
        if raw_id not in id_map:
            id_map[raw_id] = next_id
            next_id += 1

        assigned_id = id_map[raw_id]
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        label = f"{label_prefix} {assigned_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, draw_color, font_thickness)

    return frame

# === Main loop ===
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed = process_frame(frame)
    out.write(processed)

    if display_output:
        cv2.imshow("Tracking", cv2.resize(processed, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
