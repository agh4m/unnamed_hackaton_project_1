import argparse
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)


def train_yolo(params):
    # Load YOLO model
    model = YOLO('yolov5su.pt')

    # Training parameters
    epochs = params.get('num_epochs', 50)
    batch_size = params.get('batch_size', 32)
    lr = params.get('lr', 0.001)

    # Train
    model.train(
        data='data.yaml',
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        patience=params.get('patience', 10),
        save=True,
        project='yolo_runs',
        name='rf_detection'
    )

    print("Training completed. Model saved in yolo_runs/rf_detection/")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv5 for RF signal detection')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int,
                        default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')

    args = parser.parse_args()
    params = vars(args)

    train_yolo(params)


if __name__ == '__main__':
    main()
