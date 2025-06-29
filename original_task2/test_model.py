from ultralytics import YOLO

model = YOLO("..\\resources\\best.pt")
results = model.track(
    source="..\\resources\\15sec_input_720p.mp4",
    persist=True,
    conf=0.8,
    iou=0.5
)
