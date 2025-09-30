import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import scipy.ndimage
import torch.nn.functional as F
import json

# Define classes based on generated images
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

classes = ['wifi', 'bt', 'ble', 'ar_drone', 'bepop_drone', 'phantom_drone'] + \
    [name.replace(' ', '_').lower() for name in drone_spectra_classes[1:]]
NUM_CLASSES = len(classes)


class RFClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RFClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Assuming 512x512 input, after 3 pools 64x64
        self.fc1 = nn.Linear(128 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def preprocess_image(image_path):
    # Load PNG spectrogram
    img = Image.open(image_path).convert('L')  # Grayscale
    spectrogram = np.array(img) / 255.0
    # Advanced signal detection using image processing
    # Apply median filter to reduce noise
    filtered = scipy.ndimage.median_filter(spectrogram, size=3)
    blurred = scipy.ndimage.gaussian_filter(filtered, sigma=1)
    # Use 75th percentile as threshold
    threshold = np.percentile(blurred, 75)
    mask = blurred > threshold
    # Morphological operations
    mask = scipy.ndimage.binary_erosion(mask, iterations=2)
    mask = scipy.ndimage.binary_dilation(mask, iterations=3)
    # Find all signal pixels
    rows, cols = np.where(mask)
    if len(rows) > 0 and len(cols) > 0:
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        # Crop to bounding box
        spectrogram = spectrogram[rmin:rmax + 1, cmin:cmax + 1]
        # Resize back to 512x512
        spectrogram = np.array(Image.fromarray(
            (spectrogram * 255).astype(np.uint8)).resize((512, 512))) / 255.0
    else:
        # If no signal detected, resize the whole image
        spectrogram = np.array(Image.fromarray(
            (spectrogram * 255).astype(np.uint8)).resize((512, 512))) / 255.0

    # Convert to tensor
    spectrogram = torch.tensor(
        # Add batch and channel
        spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return spectrogram


def predict_custom(image_path):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RFClassifier(NUM_CLASSES)
    model.load_state_dict(torch.load('rf_classifier.pth', map_location=device))
    model.to(device)
    model.eval()

    # Preprocess image
    spectrogram = preprocess_image(image_path).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    certainty = confidence.item()

    return {"predicted_class": predicted_class, "confidence": certainty}


def main():
    parser = argparse.ArgumentParser(
        description='Predict drone type from RF spectrogram image')
    parser.add_argument('image_path', type=str,
                         help='Path to the spectrogram image')
    args = parser.parse_args()

    output = predict_custom(args.image_path)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
