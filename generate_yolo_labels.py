import os
import glob
import numpy as np
from PIL import Image
import scipy.ndimage

# 0 for background (wifi/bt/ble) others are drones
drone_spectra_classes = [
    'Background',  # 0
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

# Added later, so no background here. The incomplete classes folders refer to the dataset missing these classes, but with background.
# classes = ['wifi', 'bt', 'ble', 'ar_drone', 'bepop_drone', 'phantom_drone'] + \
#     [name.replace(' ', '_').lower() for name in drone_spectra_classes]
classes = ['ar_drone', 'bepop_drone', 'phantom_drone'] + \
    [name.replace(' ', '_').lower() for name in drone_spectra_classes]

# classes = [name.replace(' ', '_').lower() for name in drone_spectra_classes]

yolo_classes = {cls: idx for idx, cls in enumerate(classes, 0)}


def process_image(image_path):
    # Load image
    img = Image.open(image_path).convert('L')
    spectrogram = np.array(img) / 255.0

    # Apply filters as in rf_classification.py
    filtered = scipy.ndimage.median_filter(spectrogram, size=3)
    blurred = scipy.ndimage.gaussian_filter(filtered, sigma=1)
    # Use 75th percentile as threshold for "greener" parts (higher intensity)
    threshold = np.percentile(blurred, 75)
    mask = blurred > threshold
    # Morphological operations
    mask = scipy.ndimage.binary_erosion(mask, iterations=2)
    mask = scipy.ndimage.binary_dilation(mask, iterations=3)

    # Find bounding box
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None  # No signal detected

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    # Normalize to 0-1
    h, w = spectrogram.shape
    x_center = (cmin + cmax) / 2 / w
    y_center = (rmin + rmax) / 2 / h
    width = (cmax - cmin) / w
    height = (rmax - rmin) / h

    return x_center, y_center, width, height


def main():
    image_dir = 'images'
    label_dir = 'labels'
    os.makedirs(label_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    for path in image_paths:
        filename = os.path.basename(path)
        class_name = '_'.join(filename.split('_')[:-1])

        if class_name not in yolo_classes:
            continue  # Skip background or unknown

        class_id = yolo_classes[class_name]
        bbox = process_image(path)
        if bbox is None:
            continue  # No bbox found

        x, y, w, h = bbox
        label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x} {y} {w} {h}\n")

        print(f"Generated label for {filename}")


if __name__ == '__main__':
    main()
