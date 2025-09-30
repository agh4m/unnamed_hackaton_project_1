import argparse
import os
import json
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)


def predict_yolo(image_path, model_path='final_model/weights/best.pt'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Load YOLO model
    model = YOLO(model_path)

    # Run prediction
    results = model.predict(image_path, conf=0.5)

    # Extract detected classes and confidences
    detected = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            confidence = float(box.conf.item())
            detected.append({"class": cls_name, "confidence": confidence})

    return {"predicted_classes": detected}


def main():
    parser = argparse.ArgumentParser(
        description='Predict using YOLO model on a single image')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('--model', type=str, default='final_model/weights/best.pt',
                        help='Path to the trained YOLO model')
    args = parser.parse_args()

    output = predict_yolo(args.image_path, args.model)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
