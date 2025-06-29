from ultralytics import YOLO
import cv2
import time

# === Load the model ===
model = YOLO("..\\resources\\best.pt")

# === Open video ===
cap = cv2.VideoCapture("..\\resources\\15sec_input_720p.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('..\\output\\detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

total_start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, verbose=False)

    # Draw results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Detection", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

total_end = time.time()

total_duration = (total_end - total_start)/60
print(f"Total Time : {total_duration:.2f} mins")
cap.release()
out.release()
cv2.destroyAllWindows()