import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')

# Parameters
SPECTROGRAM_SIZE = (512, 512)  # h x w
FS = 125e6
NPERSEG = 1024
NOVERLAP = 512
MAX_SAMPLES = 2000000  # Limit per signal

# Folders
# SPECTROGRAM_TRAINING_PATH = 'spectrogram_training_data_20220711'
DRONE_RF_PATH = 'DroneRF'
DRONE_SPECTRA_PATH = 'DroneRFb-Spectra'
OUTPUT_FOLDER = 'images'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Drone spectra classes
drone_spectra_classes = [
    'Background',  # Wifi and BT (class 0)
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

# Function to load raw signal from .packet or .csv


def load_raw_signal(filepath):
    if filepath.endswith('.packet'):
        # Binary float32 I Q
        data = np.fromfile(filepath, dtype=np.float32)[:2*MAX_SAMPLES]
        I = data[::2]
        Q = data[1::2]
        signal = I + 1j * Q
    elif filepath.endswith('.csv'):
        with open(filepath) as f:
            line = f.read().strip()
        data = np.array(line.split(','), dtype=float)[:MAX_SAMPLES]
        I = data[::2]
        Q = data[1::2]
        signal = I + 1j * Q
    else:
        raise ValueError("Unknown file type")
    return signal

# Function to augment signal


def augment_signal(signal):
    augmented = signal.copy()
    # Random amplitude scaling
    if random.random() < 0.5:
        scale = np.random.uniform(0.5, 1.5)
        augmented *= scale
    # Add Gaussian noise
    if random.random() < 0.5:
        noise_level = np.random.uniform(0.01, 0.1) * np.max(np.abs(augmented))
        noise = np.random.normal(0, noise_level, len(
            augmented)) + 1j * np.random.normal(0, noise_level, len(augmented))
        augmented += noise
    # Frequency shift
    if random.random() < 0.3:
        delta_f = np.random.uniform(-1e6, 1e6)
        t = np.arange(len(augmented)) / FS
        augmented *= np.exp(1j * 2 * np.pi * delta_f * t)
    # Time shift
    if random.random() < 0.3:
        shift = np.random.randint(0, len(augmented))
        augmented = np.roll(augmented, shift)
    return augmented

# Function to compute spectrogram


def compute_spectrogram(signal):
    f, t, Sxx = scipy.signal.spectrogram(
        signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx = 10 * np.log10(Sxx + 1e-10)  # dB
    # Resize to SPECTROGRAM_SIZE
    img = Image.fromarray(Sxx)
    img = img.resize(SPECTROGRAM_SIZE)
    spectrogram = np.array(img)
    return spectrogram


# Collect raw files
wifi_files = []
bt_files = []
ble_files = []
ar_drone_files = []
bepop_drone_files = []
phantom_drone_files = []

# # Wifi
# wifi_files.extend(glob.glob(os.path.join(
#     SPECTROGRAM_TRAINING_PATH, 'single_packet_samples', 'wlan', '*.packet')))
# wifi_files.extend(glob.glob(os.path.join(
#     SPECTROGRAM_TRAINING_PATH, 'merged_packets', '**', '*.packet'), recursive=True))
#
# # BT
# bt_files.extend(glob.glob(os.path.join(SPECTROGRAM_TRAINING_PATH,
#                 'single_packet_samples', 'BT_classic', '*.packet')))
# bt_files.extend(glob.glob(os.path.join(SPECTROGRAM_TRAINING_PATH,
#                 'merged_packets', '**', '*.packet'), recursive=True))  # Assume some are BT
#
# # BLE
# ble_files.extend(glob.glob(os.path.join(
#     SPECTROGRAM_TRAINING_PATH, 'single_packet_samples', 'BLE_*', '*.packet')))
#
# Drones by type
ar_drone_files.extend(glob.glob(os.path.join(
    DRONE_RF_PATH, 'AR drone', '**', '*.csv'), recursive=True))
bepop_drone_files.extend(glob.glob(os.path.join(
    DRONE_RF_PATH, 'Bepop drone', '**', '*.csv'), recursive=True))
phantom_drone_files.extend(glob.glob(os.path.join(
    DRONE_RF_PATH, 'Phantom drone', '**', '*.csv'), recursive=True))

print(f"Wifi files: {len(wifi_files)}")
print(f"BT files: {len(bt_files)}")
print(f"BLE files: {len(ble_files)}")
print(f"AR Drone files: {len(ar_drone_files)}")
print(f"Bepop Drone files: {len(bepop_drone_files)}")
print(f"Phantom Drone files: {len(phantom_drone_files)}")

# Generate single spectrograms


def generate_single(files, num_samples, class_name):
    for i in range(num_samples):
        if not files:
            break
        f = random.choice(files)
        try:
            sig = load_raw_signal(f)
            sig = augment_signal(sig)
            spectrogram = compute_spectrogram(sig)
            # Normalize to 0-1
            spectrogram = (spectrogram - np.min(spectrogram)) / \
                (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
            # Invert for WiFi, BT, BLE to match drone scales
            if class_name in ['wifi', 'bt', 'ble']:
                spectrogram = 1 - spectrogram
            # Save
            filename = f"{class_name}_{i}.png"
            plt.imsave(os.path.join(OUTPUT_FOLDER, filename),
                       spectrogram, cmap='viridis')
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue


# Generate
generate_single(wifi_files, 900, 'wifi')
generate_single(bt_files, 50, 'bt')
generate_single(ble_files, 50, 'ble')
generate_single(ar_drone_files, 100, 'ar_drone')
generate_single(bepop_drone_files, 100, 'bepop_drone')
generate_single(phantom_drone_files, 100, 'phantom_drone')

# Include pre-computed drone spectra
for class_idx in range(0, 24):
    class_name = drone_spectra_classes[class_idx].replace(' ', '_').lower()
    folder = os.path.join(DRONE_SPECTRA_PATH, 'Data', str(class_idx))
    print(f"Processing class {class_idx}: {class_name}, folder: {folder}")
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, '*.npy'))
        print(f"Found {len(files)} .npy files")
        for i, f in enumerate(files):
            try:
                spec = np.load(f).astype(np.float32)
                if spec.shape != (512, 512):
                    # Resize to 512x512
                    spec = np.array(Image.fromarray(spec).resize((512, 512)))
                    print(f"Resized {f} from {spec.shape} to (512, 512)")
                filename = f"{class_name}_{i}.png"
                plt.imsave(os.path.join(OUTPUT_FOLDER, filename),
                           spec, cmap='viridis')
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Error loading {f}: {e}")
    else:
        print(f"Folder does not exist: {folder}")

print("Generation complete.")
