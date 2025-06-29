from ultralytics import YOLO
import cv2

# Load model
model = YOLO("..\\resources\\best.pt")

# Run tracking (but don’t show/save automatically)
results = model.track(
    source="..\\resources\\15sec_input_720p.mp4",
    persist=True,
    conf=0.8,
    iou=0.5,
    stream=True  # <== IMPORTANT: stream frames so you can manually process
)

# Open video writer (optional)
out = cv2.VideoWriter(".\\output\\output_1.avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))

# Iterate over each frame result
for result in results:
    frame = result.orig_img  # original frame

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            id_text = f"ID: {int(ids[i])}" if ids is not None else ""
            class_id = int(classes[i])
            label = f"{id_text}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show or write frame
    out.write(frame)  # or use cv2.imshow("Custom", frame) if you want to see it live

out.release()
