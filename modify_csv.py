import pandas as pd
import numpy as np
import motmetrics as mm

# === PATCH for NumPy 2.0+ (np.asfarray is removed) ===
import motmetrics.distances
motmetrics.distances.np.asfarray = lambda arr: np.asarray(arr, dtype=np.float64)

# === Step 1: Load and Convert (non-destructively) ===
def load_and_convert_csv(path):
    df = pd.read_csv(path)
    df = df[['frame_id', 'id', 'x1', 'y1', 'x2', 'y2', 'confidence']]

    # Convert corner coords to width/height
    df['x'] = df['x1']
    df['y'] = df['y1']
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']

    # Rename for motmetrics
    df = df.rename(columns={
        'frame_id': 'FrameId',
        'id': 'Id'
    })

    return df[['FrameId', 'Id', 'x', 'y', 'w', 'h']]

# === Step 2: Convert to MOTAccumulator ===
def to_accumulator(gt_df, pred_df):
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in sorted(gt_df['FrameId'].unique()):
        gt_frame = gt_df[gt_df['FrameId'] == frame]
        pred_frame = pred_df[pred_df['FrameId'] == frame]

        gt_ids = gt_frame['Id'].tolist()
        pred_ids = pred_frame['Id'].tolist()

        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    return acc

# === Step 3: Load both CSVs ===
gt_df = load_and_convert_csv(".\\original_task2\\output\\manual_id_output.csv")
pred_df = load_and_convert_csv(".\\task2\\output\\identify.csv")

# === Step 4: Evaluate ===
acc = to_accumulator(gt_df, pred_df)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='Comparison')

# === Step 5: Print results ===
print(mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=None
))
