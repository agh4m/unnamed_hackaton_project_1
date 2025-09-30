import os
import glob
import json
from PIL import Image
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

# Same classes as in generate_yolo_labels.py
drone_spectra_classes = [
    'Background',
    'DJI Phantom 3',
    'DJI Phantom 4 Pro',
    'DJI MATRICE 200',
    'DJI MATRICE 100',
    'DJI Air 2S',
    'DJI Mini 3 Pro',
    'DJI Inspire 2',
    'DJI Mavic Pro',
    'DJI Mini 2',
    'DJI Mavic 3',
    'DJI MATRICE 300',
    'DJI Phantom 4 Pro RTK',
    'DJI MATRICE 30T',
    'DJI AVATA',
    'DJI DIY',
    'DJI MATRICE 600 Pro',
    'VBar',
    'FrSky X20',
    'Futuba T16IZ',
    'Taranis Plus',
    'RadioLink AT9S',
    'Futaba T14SG',
    'Skydroid'
]

# classes = ['wifi', 'bt', 'ble', 'ar_drone', 'bepop_drone', 'phantom_drone'] + \
#     [name.replace(' ', '_').lower() for name in drone_spectra_classes]
classes = ['ar_drone', 'bepop_drone', 'phantom_drone'] + \
    [name.replace(' ', '_').lower() for name in drone_spectra_classes]

yolo_classes = {cls: idx for idx, cls in enumerate(classes, 0)}


def generate_labels_from_model(image_dir, label_dir, model_path, conf_threshold=0.5, low_conf_threshold=0.7):
    os.makedirs(label_dir, exist_ok=True)

    # Load the trained YOLO model
    model = YOLO(model_path)

    low_conf_detections = {}

    image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    for path in image_paths:
        filename = os.path.basename(path)
        label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

        # Run inference
        results = model.predict(path, conf=conf_threshold)

        with open(label_path, 'w') as f:
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = classes[cls_id]
                    confidence = float(box.conf.item())
                    # Get bounding box in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Get image dimensions
                    img = Image.open(path)
                    w, h = img.size  # PIL gives (width, height)

                    # Convert to YOLO format (normalized)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    # Write to label file
                    f.write(f"{cls_id} {x_center:.6f} {
                            y_center:.6f} {width:.6f} {height:.6f}\n")

                    # Check for low confidence
                    if confidence < low_conf_threshold:
                        if filename not in low_conf_detections:
                            low_conf_detections[filename] = []
                        low_conf_detections[filename].append({
                            "class": cls_name,
                            "confidence": confidence,
                            "bbox": [x_center, y_center, width, height]
                        })

        print(f"Generated label for {filename}")

    # Save low confidence detections to JSON
    with open('low_confidence_detections.json', 'w') as f:
        json.dump(low_conf_detections, f, indent=4)

    print("Low confidence detections saved to low_confidence_detections.json")


def main():
    image_dir = 'images'
    label_dir = 'labels'
    # Update to your trained model path
    model_path = 'yolo_runs/rf_detection4/weights/best.pt'

    generate_labels_from_model(image_dir, label_dir, model_path)


if __name__ == '__main__':
    main()
