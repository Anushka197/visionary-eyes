from ultralytics import YOLO
import pandas as pd
import os

# === Load model and track
model = YOLO("..\\resources\\best.pt")

results = model.track(
    source="..\\resources\\15sec_input_720p.mp4",
    persist=True,
    conf=0.4,
    iou=0.5,
    save=True
)

# === Prepare to save data
all_data = []

# === Extract frame-wise tracking data
for r in results:
    frame_num = r.path.split("frame")[-1].split(".")[0] if "frame" in r.path else "?"
    if r.boxes is None:
        continue
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names.get(cls, f"class_{cls}")
        all_data.append({
            "frame": int(frame_num) if frame_num != "?" else -1,
            "id": track_id,
            "class": class_name,
            "conf": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

# === Create DataFrame and save
df = pd.DataFrame(all_data)
csv_path = "..\\output\\id_stats.csv"
df.to_csv(csv_path, index=False)
print(f"\nTracking data saved to: {csv_path}")

# === Analyze
print("\nOBJECT COUNT PER CLASS:")
print(df['class'].value_counts())

print("\nAVERAGE CONFIDENCE PER CLASS:")
print(df.groupby('class')['conf'].mean().round(3))

print("\nFRAME COUNT PER OBJECT ID:")
print(df.groupby('id')['frame'].count().sort_values(ascending=False).head(10))

# === BONUS: Movement range per ID
print("\nMOVEMENT RANGE PER OBJECT ID (X range):")
df['x_center'] = (df['x1'] + df['x2']) / 2
x_range = df.groupby('id')['x_center'].agg(['min', 'max'])
x_range['range'] = (x_range['max'] - x_range['min']).round(1)
print(x_range[['range']].sort_values(by='range', ascending=False).head(10))
