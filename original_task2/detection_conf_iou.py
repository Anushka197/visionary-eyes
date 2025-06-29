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

start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(
        source=frame,
        conf=0.8,    # confidence threshold
        iou=0.5,      # NMS IoU threshold
        verbose=False
    )


    # Draw results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            iou = float(box.iou[0]) if hasattr(box, 'iou') else 0.5  # if iou info is present

            # label_name = model.names[cls] if cls in model.names else f"Class {cls}"
            label = f"conf={conf:.2f} | iou={iou:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("Detection", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()
total = (end - start)/60
print(f"Total Time : {total:.2f} mins")

cap.release()
out.release()
cv2.destroyAllWindows()
