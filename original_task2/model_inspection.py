from ultralytics import YOLO
import torch

# === Load your custom model ===
model = YOLO("..\\resources\\best.pt")

print("\nMODEL SUMMARY")
print("="*50)
print(model)                          # Overall model summary
print("\nModel Architecture Layers:")
print(model.model)                    # Full architecture

print("\nPARAMETER DETAILS")
print("="*50)
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print(f"Total Params: {total_params:,}")
print(f"Trainable Params: {trainable_params:,}")

print("\nMODEL TASK TYPE")
print("="*50)
print(f"Task: {model.task}")          # detect / segment / pose / classify

print("\nCLASS LABELS")
print("="*50)
if hasattr(model, 'names'):
    for cls_id, name in model.names.items():
        print(f"{cls_id}: {name}")
else:
    print("No class names found.")

print("\nYAML / CONFIG SETTINGS")
print("="*50)
print(model.yaml)                     # Basic training settings
print("\nOverrides (inference-time options):")
print(model.overrides)

print("\nSTRIDE AND ANCHORS")
print("="*50)
if hasattr(model.model, 'stride'):
    print(f"Strides: {model.model.stride}")
if hasattr(model.model, 'anchors'):
    print(f"Anchors: {model.model.anchors}")

print("\nINPUT REQUIREMENTS")
print("="*50)
imgsz = model.overrides.get('imgsz', [640, 640])
print(f"Expected image size: {imgsz}")
print("Supported input size multiple:", model.model.stride if hasattr(model.model, 'stride') else "Unknown")

print("\nTRACKING CAPABILITY CHECK")
print("="*50)
try:
    print("Testing .track() on dummy input...")
    results = model.track(source="https://ultralytics.com/images/bus.jpg", conf=0.25, iou=0.5, stream=True)
    print(".track() is supported. Model works with Ultralytics tracking.")
except Exception as e:
    print(".track() failed:", str(e))

print("\nMODULE NAMES & SHAPES")
print("="*50)
for name, param in model.model.named_parameters():
    print(f"{name:60s}  | shape: {tuple(param.shape)}")

print("\nDEVICE & TORCH INFO")
print("="*50)
print("PyTorch version:", torch.__version__)
print("Device:", model.device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nINSPECTION COMPLETE.")
