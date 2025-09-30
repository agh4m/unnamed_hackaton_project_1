This repo includes my submission for the [] Hackathon 2025.

Included are:
  - A script to prepare the dataset, and convert them to usable images
  - A script to generate labels for the dataset
  - A script to preprocess the dataset
  - A script to train a model on the dataset
  - A script to evaluate the model on a test set
  - A script to batch test the model on multiple images

The model is based on a pre-trained YOLOv5 model, fine-tuned on the following datasets:
  - [DroneRF](https://github.com/Al-Sad/DroneRF)
  - [DroneRFb-Spectra](https://ieee-dataport.org/documents/dronerfb-spectra-rf-spectrogram-dataset-drone-recognition)

It is expected that these datasets are available on the root directory of the repo for the scripts to work.

The model is trained on a mix of drone models from both datasets, as well as some background signals from the DroneRFb-Spectra dataset.

Also included is a custom model, to serve as experimentation for what a simple CNN model could achieve on this task.
Results for both are included in their respective folders.

Finally, a web gui is included to easily visualise the results of the model on a given image, as well as to provide a Human in the middle feedback loop for further training the model.
