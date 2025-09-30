from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import time
import json
import subprocess
from custom_model.predict import predict_custom
from predict_yolo import predict_yolo

app = Flask(__name__)
CORS(app)

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


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    model_type = request.form.get('modelType')

    if not model_type:
        return jsonify({'error': 'modelType is required'}), 400

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_path = temp_file.name
        file.save(temp_path)

    try:
        # Extract real class from filename
        filename = file.filename
        real_class = '_'.join(filename.split('_')[:-1])

        start_time = time.time()
        if model_type == 'custom':
            prediction = predict_custom(temp_path)
        elif model_type == 'yolo':
            prediction = predict_yolo(
                temp_path, model_path='final_model/weights/best.pt')
        else:
            return jsonify({'error': 'Invalid modelType'}), 400
        processing_time = time.time() - start_time

        print(f"Returning: realClass={real_class}, prediction={
              prediction}, time={processing_time * 1000:.0f}ms")
        return jsonify({
            'realClass': real_class,
            'processingTime': round(processing_time * 1000),
            **prediction
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        os.unlink(temp_path)


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('images', filename)


@app.route('/discover_low_confidence', methods=['GET'])
def discover_low_confidence():
    low_conf_file = 'low_confidence_detections.json'
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'

    if force_refresh:
        print("Force regenerating low confidence detections...")
        try:
            subprocess.run(
                ['python', 'generate_labels_from_model.py'], check=True)
            print("Regeneration completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error regenerating: {e}")
            return jsonify({'error': 'Failed to regenerate low confidence detections'}), 500

    if not os.path.exists(low_conf_file):
        return jsonify({'error': f'Low confidence detections file "{low_conf_file}" not found. Run generate_labels_from_model.py first.'}), 404

    try:
        with open(low_conf_file, 'r') as f:
            low_conf_data = json.load(f)

        low_conf_images = []
        for filename, detections in low_conf_data.items():
            if detections:
                max_conf = max(d['confidence'] for d in detections)
                # Include bbox for overlay
                formatted_detections = [
                    {'class': d['class'], 'confidence': d['confidence'], 'bbox': d['bbox']} for d in detections]
                low_conf_images.append({
                    'filename': filename,
                    'max_confidence': max_conf,
                    'detections': formatted_detections
                })
            else:
                # No low conf detections
                low_conf_images.append({
                    'filename': filename,
                    'max_confidence': 0.0,
                    'detections': []
                })

        result = {'low_confidence_images': low_conf_images}
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/save_labels', methods=['POST'])
def save_labels():
    print("Received save_labels request")
    data = request.get_json()
    print(f"Data: {data}")
    if not data or 'filename' not in data or 'detections' not in data:
        print("Invalid data")
        return jsonify({'error': 'Invalid data. Need filename and detections.'}), 400

    filename = data['filename']
    detections = data['detections']

    label_path = os.path.join('labels', filename.replace('.png', '.txt'))
    print(f"Saving to {label_path}")

    try:
        with open(label_path, 'w') as f:
            for det in detections:
                cls_name = det['class']
                bbox = det['bbox']
                print(f"Processing {cls_name}, bbox {bbox}")
                if cls_name not in yolo_classes:
                    print(f"Unknown class: {cls_name}")
                    return jsonify({'error': f'Unknown class: {cls_name}'}), 400
                class_id = yolo_classes[cls_name]
                x, y, w, h = bbox
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Remove from low confidence detections
        low_conf_file = 'low_confidence_detections.json'
        if os.path.exists(low_conf_file):
            try:
                with open(low_conf_file, 'r') as f:
                    low_conf_data = json.load(f)
                if filename in low_conf_data:
                    del low_conf_data[filename]
                    with open(low_conf_file, 'w') as f:
                        json.dump(low_conf_data, f, indent=4)
                    print(f"Removed {filename} from low confidence detections")
            except Exception as e:
                print(f"Error updating low_conf_data: {e}")

        print("Labels saved successfully")
        return jsonify({'message': 'Labels saved successfully'})
    except Exception as e:
        print(f"Error saving labels: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
