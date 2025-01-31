from ultralytics import YOLO

model = YOLO('yolo11x.pt')

train_results = model.train(
    data="yolo.yaml",  # path to dataset YAML
    epochs=3,  # number of training epochs
    imgsz=256,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model

print(f"Saved model to {path}")