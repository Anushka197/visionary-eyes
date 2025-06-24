import cv2
import numpy as np
import csv
from ultralytics import YOLO
from collections import defaultdict
from scipy.spatial.distance import cosine
import os

# Load YOLOv11 model
model = YOLO('../best.pt')

# Settings
CONFIDENCE_THRESHOLD = 0.4
N_FRAMES = 300
COLOR_HIST_BINS = [8, 8, 8]
OUTPUT_CSV = '../tacticam_player_ids.csv'
OUTPUT_VIDEO = '../tacticam_annotated.mp4'

# Helper: Extract HSV histogram
def get_color_histogram(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    player_img = frame[y1:y2, x1:x2]
    if player_img.size == 0:
        return np.zeros((np.prod(COLOR_HIST_BINS),))
    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, COLOR_HIST_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Run detection
def detect_players(video_path):
    cap = cv2.VideoCapture(video_path)
    detections = defaultdict(list)
    frame_count = 0

    while cap.isOpened() and frame_count < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > CONFIDENCE_THRESHOLD and cls == 0:
                bbox = box.xyxy[0].cpu().numpy()
                hist = get_color_histogram(frame, bbox)
                detections[frame_count].append({'bbox': bbox, 'hist': hist})

        frame_count += 1

    cap.release()
    return detections

# Matching based on histogram similarity
def match_players(broadcast_data, tacticam_data):
    player_id_counter = 0
    broadcast_embeddings = []

    for frame in broadcast_data.values():
        for detection in frame:
            broadcast_embeddings.append(detection['hist'])

    assigned_data = []

    for frame_num, detections in tacticam_data.items():
        for det in detections:
            hist = det['hist']
            similarities = [1 - cosine(hist, emb) for emb in broadcast_embeddings]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            if best_score > 0.8:
                matched_id = best_idx
            else:
                matched_id = len(broadcast_embeddings)
                broadcast_embeddings.append(hist)

            det['id'] = matched_id
            assigned_data.append({
                'frame': frame_num,
                'bbox': det['bbox'],
                'id': matched_id
            })

    return assigned_data

# Save to CSV
def save_to_csv(data, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'player_id'])
        for item in data:
            x1, y1, x2, y2 = item['bbox']
            writer.writerow([item['frame'], int(x1), int(y1), int(x2), int(y2), item['id']])

# Visualize and save video
def visualize_tracking(video_path, tracked_data, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    data_by_frame = defaultdict(list)
    for d in tracked_data:
        data_by_frame[d['frame']].append(d)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > N_FRAMES:
            break

        if frame_idx in data_by_frame:
            for d in data_by_frame[frame_idx]:
                x1, y1, x2, y2 = map(int, d['bbox'])
                pid = d['id']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {pid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

# Main
if __name__ == "__main__":
    print("[1] Detecting players in broadcast...")
    broadcast_data = detect_players('../broadcast.mp4')

    print("[2] Detecting players in tacticam...")
    tacticam_data = detect_players('../tacticam.mp4')

    print("[3] Mapping player IDs...")
    assigned_data = match_players(broadcast_data, tacticam_data)

    print(f"[4] Saving to CSV: {OUTPUT_CSV}")
    save_to_csv(assigned_data, OUTPUT_CSV)

    print(f"[5] Visualizing tracking and saving video: {OUTPUT_VIDEO}")
    visualize_tracking('tacticam.mp4', assigned_data, OUTPUT_VIDEO)

    print("✅ Done! CSV and annotated video saved.")
