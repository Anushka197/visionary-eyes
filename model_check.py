from ultralytics import YOLO

model = YOLO("../best.pt")

# Print basic info
# print("Model type:", model.model.model.__class__)
# print("Model names (classes):", model.names)
# print(model.model)
import pprint
pprint.pprint(model.model.yaml)
