import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLOv11 model
model = YOLO("../best.pt")  # Replace with your model path

# Initialize video capture
cap = cv2.VideoCapture("../15sec_input_720p.mp4")  # Or use 0 for webcam

# Store tracked players
player_id_counter = 0
tracked_players = {}  # {id: {'bbox': [x1,y1,x2,y2], 'features': ..., 'frames': last_seen}}

# Function to compute basic appearance feature (color histogram)
def get_features(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# IOU matcher
def iou(boxA, boxB):
    xa, ya, xa2, ya2 = boxA
    xb, yb, xb2, yb2 = boxB
    inter_x1 = max(xa, xb)
    inter_y1 = max(ya, yb)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    boxA_area = (xa2 - xa) * (ya2 - ya)
    boxB_area = (xb2 - xb) * (yb2 - yb)
    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-5)

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1

    detections = model(frame, verbose=False)[0]
    new_tracks = []
    for i, det in enumerate(detections.boxes.data):
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) != 0:  # Assuming class 0 is 'player'
            continue
        bbox = [x1, y1, x2, y2]
        features = get_features(frame, bbox)
        matched_id = None

        # Try to match with existing players
        for pid, pdata in tracked_players.items():
            if iou(bbox, pdata['bbox']) > 0.4:
                matched_id = pid
                break
            sim = np.dot(pdata['features'], features)
            if sim > 0.9:
                matched_id = pid
                break

        if matched_id is None:
            matched_id = player_id_counter
            player_id_counter += 1

        tracked_players[matched_id] = {'bbox': bbox, 'features': features, 'frames': frame_count}
        new_tracks.append((matched_id, bbox))

    # Draw
    for pid, bbox in new_tracks:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Re-ID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
