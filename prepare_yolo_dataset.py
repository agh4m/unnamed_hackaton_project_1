import os
import shutil
from sklearn.model_selection import train_test_split


def main():
    image_dir = 'images'
    label_dir = 'labels'
    dataset_dir = 'yolo_dataset'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    train_images = os.path.join(train_dir, 'images')
    train_labels = os.path.join(train_dir, 'labels')
    val_images = os.path.join(val_dir, 'images')
    val_labels = os.path.join(val_dir, 'labels')

    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    # Get all image files that have labels
    image_files = []
    for f in os.listdir(image_dir):
        if f.endswith('.png'):
            label_f = f.replace('.png', '.txt')
            if os.path.exists(os.path.join(label_dir, label_f)):
                image_files.append(f)

    # Split 80/20
    train_files, val_files = train_test_split(
        image_files, test_size=0.2, random_state=42)

    # Copy to train
    for f in train_files:
        shutil.copy(os.path.join(image_dir, f), os.path.join(train_images, f))
        label_f = f.replace('.png', '.txt')
        shutil.copy(os.path.join(label_dir, label_f),
                    os.path.join(train_labels, label_f))

    # Copy to val
    for f in val_files:
        shutil.copy(os.path.join(image_dir, f), os.path.join(val_images, f))
        label_f = f.replace('.png', '.txt')
        shutil.copy(os.path.join(label_dir, label_f),
                    os.path.join(val_labels, label_f))

    # Create data.yaml
    data_yaml = f"""
train: {os.path.abspath(train_images)}
val: {os.path.abspath(val_images)}

nc: {len(yolo_classes)}  # number of classes
names: {list(yolo_classes.keys())}  # class names
"""

    with open('data.yaml', 'w') as f:
        f.write(data_yaml)

    print("Dataset prepared.")


if __name__ == '__main__':
    # Define yolo_classes as in generate_yolo_labels.py
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

    yolo_classes = {cls: idx for idx, cls in enumerate(
        classes, 0)}

    main()
