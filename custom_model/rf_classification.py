from tqdm import tqdm
# import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import scipy.signal
import scipy.ndimage
# import matplotlib
# matplotlib.use('Agg')

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
    [name.replace(' ', '_').lower() for name in drone_spectra_classes]
NUM_CLASSES = len(classes)

# Data augmentation functions for spectrograms


def augment_spectrogram(spectrogram, params):
    # Add noise
    noise_level = params.get('noise_level', 0.1)
    noise = np.random.normal(0, noise_level, spectrogram.shape)
    spectrogram = spectrogram + noise
    # Random crop or shift
    if np.random.rand() > 0.5:
        shift_max = params.get('shift_max', 50)
        shift = np.random.randint(-shift_max, shift_max)
        spectrogram = np.roll(spectrogram, shift, axis=1)
    # Normalize
    spectrogram = (spectrogram - np.min(spectrogram)) / \
        (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
    return spectrogram

# For raw RF signals (if needed for DroneRF)


def load_raw_signal(filepath):
    print(f"Loading raw signal from {filepath}")
    with open(filepath) as f:
        line = f.read().strip()
    data = np.array(line.split(','), dtype=float)[
        :1000000]  # Limit to 1M values
    I = data[::2]
    Q = data[1::2]
    signal = I + 1j * Q
    return signal


def compute_spectrogram(signal, fs=1e6, nperseg=1024, noverlap=512):
    f, t, Sxx = scipy.signal.spectrogram(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx + 1e-10)  # dB
    return Sxx


def augment_signal(signal):
    # Add noise
    noise_level = 0.1
    noise = noise_level * (np.random.randn(len(signal)) +
                           1j * np.random.randn(len(signal)))
    signal = signal + noise
    # Frequency shift
    delta_f = np.random.uniform(-1e4, 1e4)  # 10kHz shift
    t = np.arange(len(signal)) / 1e6
    signal = signal * np.exp(1j * 2 * np.pi * delta_f * t)
    # Time shift
    shift = np.random.randint(-1000, 1000)
    signal = np.roll(signal, shift)
    return signal

# Dataset class


class RFDataset(Dataset):
    def __init__(self, paths, labels, augment=True, params=None):
        self.paths = paths
        self.labels = labels
        self.augment = augment
        self.params = params or {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        if path.endswith('.png'):
            # Load PNG spectrogram
            img = Image.open(path).convert('L')  # Grayscale
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
                # Resize back to 224x224
                spectrogram = np.array(Image.fromarray(
                    (spectrogram * 255).astype(np.uint8)).resize((224, 224))) / 255.0
        elif path.endswith('.npy'):
            # Load NPY spectrogram
            spectrogram = np.load(path).astype(np.float32)
            # Resize if needed
            if spectrogram.shape != (224, 224):
                spectrogram = np.array(Image.fromarray(
                    spectrogram).resize((224, 224)))
            spectrogram = (spectrogram - np.min(spectrogram)) / \
                (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
        else:
            # Assume raw signal, compute spectrogram
            signal = load_raw_signal(path)
            if self.augment:
                signal = augment_signal(signal)
            spectrogram = compute_spectrogram(signal)
            # Already resized in compute_spectrogram? No, in the code above it's after compute
            # spectrogram is 2D, resize
            h, w = spectrogram.shape
            if h != 224 or w != 224:
                spectrogram = np.array(Image.fromarray(
                    spectrogram).resize((224, 224)))

        if self.augment:
            spectrogram = augment_spectrogram(spectrogram, self.params)

        # Convert to tensor
        spectrogram = torch.tensor(
            spectrogram, dtype=torch.float32).unsqueeze(0)  # Add channel
        label = torch.tensor(label, dtype=torch.long)
        return spectrogram, label

# CNN Model


class RFClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(RFClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Modify first conv layer for 1 channel (grayscale)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        # Replace fc layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)
        return x

# Training


def train_model(params):
    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Collect data from generated images
    data_paths = glob.glob(os.path.join('images', '*.png'))
    print(f"Current working directory: {os.getcwd()}")
    print(f"Found {len(data_paths)} png files in images/")
    if data_paths:
        print(f"Sample paths: {data_paths[:3]}")
    valid_paths = []
    valid_labels = []
    class_name = None
    path = None

    for path in data_paths:
        filename = os.path.basename(path)
        # e.g., 'wifi' or 'dji_phantom_3'
        class_name = '_'.join(filename.split('_')[:-1])

        if ((class_name in classes) and (class_name not in ['wifi', 'bt', 'ble'])):
            valid_paths.append(path)
            valid_labels.append(classes.index(class_name))

    data_paths = valid_paths
    print(f"Valid images after filtering: {len(data_paths)}")
    labels = valid_labels

    print(f"Valid images: {len(valid_paths)}")
    if valid_paths:
        print(f"Sample classes: {
              ['_'.join(os.path.basename(p).split('_')[:-1]) for p in valid_paths[:5]]}")

    # Sample to balance classes
    max_samples_per_class = params.get('max_samples', 100)
    sampled_paths = []
    sampled_labels = []
    for cls in range(NUM_CLASSES):
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        sample_size = min(len(cls_indices), max_samples_per_class)
        if sample_size > 0:
            sampled_indices = np.random.choice(
                cls_indices, size=sample_size, replace=False)
            sampled_paths.extend([data_paths[i] for i in sampled_indices])
            sampled_labels.extend([labels[i] for i in sampled_indices])

    data_paths = sampled_paths
    labels = sampled_labels

    print(f"Total samples after sampling: {len(data_paths)}")
    print(f"Labels distribution after sampling: {np.bincount(labels)}")

    # Use k-fold cross-validation
    from sklearn.model_selection import StratifiedKFold
    k = params.get('k', 5)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    num_epochs = params.get('num_epochs', 50)
    patience = params.get('patience', 10)
    batch_size = params.get('batch_size', 32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels)):
        print(f"Fold {fold + 1}/{k}")

        train_paths_fold = [data_paths[i] for i in train_idx]
        train_labels_fold = [labels[i] for i in train_idx]
        val_paths_fold = [data_paths[i] for i in val_idx]
        val_labels_fold = [labels[i] for i in val_idx]

        train_dataset = RFDataset(
            train_paths_fold, train_labels_fold, augment=True, params=params)
        val_dataset = RFDataset(
            val_paths_fold, val_labels_fold, augment=False, params=params)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        model = RFClassifier(NUM_CLASSES, dropout=params.get('dropout', 0.5))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params.get(
            'lr', 0.001), weight_decay=params.get('wd', 1e-4))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.get(
            'step_size', 10), gamma=params.get('gamma', 0.1))

        model.to(device)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f'Fold {
                             fold + 1}, Epoch {epoch + 1} Training')
            for inputs, targets in train_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())
            train_loss /= len(train_loader)
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            val_bar = tqdm(val_loader, desc=f'Fold {
                           fold + 1}, Epoch {epoch + 1} Validation')
            with torch.no_grad():
                for inputs, targets in val_bar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
            val_loss /= len(val_loader)
            accuracy = correct / len(val_dataset)
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {
                  train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {
                          epoch + 1} for fold {fold + 1}")
                    break

            scheduler.step()

        # After training fold, evaluate on val set
        fold_accuracy = correct / len(val_dataset)  # Last accuracy
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} final accuracy: {fold_accuracy:.4f}")

    # Average accuracy across folds
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Average cross-validation accuracy: {avg_accuracy:.4f}")

    return avg_accuracy

    # Loss curve - commented out for testing
    # epochs_trained = len(train_losses)
    # plt.figure()
    # plt.plot(range(1, epochs_trained + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, epochs_trained + 1), val_losses, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('loss_curve.png')
    # plt.close()
    # print("Loss curve saved to loss_curve.png")

    # Confusion matrix - commented out for testing
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # model.eval()
    # all_preds = []
    # all_labels = []
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         all_preds.extend(predicted.cpu().numpy())
    #         all_labels.extend(targets.cpu().numpy())
    # cm = confusion_matrix(all_labels, all_preds)
    # # Get unique labels present in test
    # unique_labels = sorted(set(all_labels))
    # display_labels = [classes[i] for i in unique_labels]
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm, display_labels=display_labels)
    # disp.plot()
    # plt.savefig('confusion_matrix.png')
    # plt.close()
    # print("Confusion matrix saved to confusion_matrix.png")


if __name__ == '__main__':
    train_model({})
