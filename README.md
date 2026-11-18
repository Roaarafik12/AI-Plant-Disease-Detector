# AI Plant Disease Detector 🌿

An image-classification project that uses deep learning to identify plant diseases from leaf images. The model is built with **TensorFlow/Keras** and **MobileNetV2** using transfer learning, and is designed to run in **Google Colab**.

## Project Overview

The notebook downloads the PlantVillage dataset, prepares the images, trains and evaluates a deep-learning model, and allows the user to upload a new leaf image to receive a predicted class and confidence score.

### Main Features

- Loads the **PlantVillage** dataset using TensorFlow Datasets.
- Resizes images to **224 × 224** pixels.
- Uses data augmentation (random horizontal flip, rotation, and zoom).
- Builds a transfer-learning model based on **MobileNetV2** with ImageNet weights.
- Trains the classification head and evaluates the model on a validation split.
- Displays training/validation accuracy and loss curves.
- Accepts a user-uploaded leaf image and predicts its class.
- Saves the trained model as `plant_disease_model.keras` and class labels as `class_names.json`.

## Technologies Used

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pillow (PIL)
- Google Colab

## Repository Structure

```text
AI-Plant-Disease-Detector/
│
├── plant_disease_model.ipynb   # Main notebook: data loading, training, evaluation, prediction
├── README.md                   # Project documentation
├── plant_disease_model.keras   # Generated trained model (created after running the notebook)
└── class_names.json            # Generated class-label mapping (created after running the notebook)
```

> The `.keras` model and `class_names.json` are generated when the notebook runs. They are not included automatically when you upload the notebook to GitHub. Add them to the repository only if you want to share the trained model and its labels.

## Getting Started

### Option 1: Run in Google Colab (Recommended)

1. Open `plant_disease_model.ipynb` from this repository.
2. Select **Open in Colab** (or upload the notebook to Google Colab).
3. Run the notebook cells in order.
4. Allow the notebook to install the required libraries and download the PlantVillage dataset.
5. Wait for training and evaluation to finish.
6. When prompted, upload an image of a plant leaf to test the model.
7. The notebook will display the predicted class and confidence score, then save and offer the trained model for download.

**Note:** Dataset download and model training may take some time. Enabling a GPU runtime in Google Colab can help speed up training.

### Option 2: Run Locally

A Google Colab environment is used in the current notebook. Running it locally requires adapting the Google Colab file-upload and download commands.

Install the main dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib pillow
```

Then open the notebook in Jupyter and run the cells after replacing the `google.colab.files` upload/download workflow with local file paths.

## Model Architecture

The project uses **MobileNetV2** as a pre-trained feature extractor:

1. Input image: `224 × 224 × 3`.
2. Data augmentation: random flip, rotation, and zoom.
3. Rescaling: pixel values are mapped to the range `[-1, 1]`.
4. MobileNetV2: ImageNet pre-trained weights, with the base model frozen during training.
5. Global Average Pooling.
6. Dropout (`0.3`).
7. Dense output layer with softmax activation for classification.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as an evaluation metric.

## Dataset

This project uses the **PlantVillage** dataset through TensorFlow Datasets (`plant_village`).

The notebook creates training and validation subsets from the dataset's training split:

- Training: 80%
- Validation: 20%

The class names are read from the dataset metadata, so the output layer is configured according to the number of available classes.

Please review and follow the dataset's own license and usage terms before redistributing the dataset or derived materials.

## Output Files

After training, the notebook generates:

- `plant_disease_model.keras` — the trained Keras model.
- `class_names.json` — the list of class labels used to interpret model predictions.

Keep the model and class-label file together when sharing or reusing the trained model.

## Important Notes & Limitations

- The prediction is an **AI-generated estimate**, not a confirmed agricultural diagnosis.
- Results depend on image quality, lighting, leaf condition, and how closely an uploaded image resembles the training data.
- The validation accuracy reported by the notebook should be measured by running the notebook; no fixed accuracy is claimed here.
- The current notebook uses a frozen MobileNetV2 base model and trains the added classification layers for 8 epochs.

## Future Improvements

- Add a separate inference script or a simple web interface.
- Evaluate the model on real-world leaf images outside the dataset.
- Add precision, recall, F1-score, and a confusion matrix.
- Experiment with fine-tuning MobileNetV2 and different training settings.
- Improve robustness to different lighting, backgrounds, and image conditions.

## Disclaimer

This project is intended for educational and experimental purposes. Do not use its predictions as the sole basis for plant treatment or agricultural decisions.

## Author

**[Your Name]**

If you find this project useful, feel free to ⭐ the repository.
