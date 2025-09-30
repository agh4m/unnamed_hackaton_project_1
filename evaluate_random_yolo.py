import argparse
import os
import random
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)


def get_true_class(filename):
    class_name = '_'.join(filename.split('_')[:-1])
    return class_name


def evaluate_random_yolo(model_path, n):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return

    # Load YOLO model
    model = YOLO(model_path)

    # Get list of images
    images_dir = 'images'
    if not os.path.exists(images_dir):
        print(f"Images directory '{images_dir}' not found.")
        return
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if len(image_files) < n:
        print(f"Only {len(image_files)} images available, selecting all.")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, n)

    correct = 0
    total = len(selected_images)

    for img_file in selected_images:
        img_path = os.path.join(images_dir, img_file)
        true_class = get_true_class(img_file)

        # Run prediction
        results = model.predict(img_path, conf=0.5)

        # Extract detected classes
        detected_classes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = result.names[cls_id]
                detected_classes.append(cls_name)

        # Check if true class is detected
        if true_class in detected_classes:
            correct += 1

        print(f"Image: {img_file}")
        print(f"True class: {true_class}")
        print(f"Detected classes: {detected_classes}")
        print(f"Match: {true_class in detected_classes}")
        print("---")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Evaluated {total} images")
    print(f"Correct detections: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO model on n random images from images/ folder')
    parser.add_argument(
        'n', type=int, help='Number of random images to evaluate')
    parser.add_argument('--model', type=str, default='final_model/weights/best.pt',
                        help='Path to the trained YOLO model')
    args = parser.parse_args()

    evaluate_random_yolo(args.model, args.n)


if __name__ == '__main__':
    main()
