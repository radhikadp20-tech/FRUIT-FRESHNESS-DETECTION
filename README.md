

An image classification project that detects whether a fruit is **Fresh** or **Rotten** using deep learning. The project compares a custom CNN approach with a fine-tuned **MobileNetV2** model and provides an interactive **Streamlit** interface for making predictions.

## 📌 Project Overview

Fruit freshness classification can be useful in applications such as food quality inspection, retail, supply-chain management, and automated sorting systems.

This project uses deep learning-based image classification to identify the freshness condition of fruits from images.

### Key Features

* 🍎 Fruit freshness classification
* 🧠 Deep learning-based image classification
* 📱 MobileNetV2 transfer learning
* 🔬 Custom CNN baseline model
* 📊 Model performance evaluation
* 📈 Training/validation curves
* 📉 Confusion matrix
* 🖼️ Sample predictions
* 🌐 Interactive Streamlit application

---

## 🏗️ Project Workflow

```text
Fruit Image
     ↓
Image Preprocessing
     ↓
CNN / MobileNetV2 Model
     ↓
Feature Extraction & Classification
     ↓
Fresh / Rotten Prediction
     ↓
Streamlit Interface
```

---

## 🤖 Models Used

### 1. Custom CNN

A Convolutional Neural Network was developed as a baseline model for fruit freshness classification.

The CNN learns visual features such as:

* Shape
* Texture
* Color
* Surface patterns

and uses these features to classify fruit images.

### 2. MobileNetV2

MobileNetV2 was used with **transfer learning** to improve classification performance while keeping the model relatively lightweight.

The pretrained network provides useful visual features, which are then fine-tuned for the fruit freshness classification task.

---

## 📊 Model Evaluation

The project includes several evaluation outputs:

* Class distribution
* Training and validation curves
* Confusion matrix
* Sample images
* Sample predictions

These visualizations help evaluate model learning, classification performance, and possible overfitting.

---

## 🖥️ Streamlit Application

The project includes a Streamlit application that allows users to interact with the trained model.

The application can:

1. Accept a fruit image.
2. Process the image.
3. Pass it through the trained MobileNetV2 model.
4. Generate a freshness prediction.
5. Display the prediction to the user.

---

## 📁 Project Structure

```text
FRUIT FRESHNESS/
│
├── app.py
├── fruit_freshness_mobilenetv2.h5
├── fruit_freshness_cnn (1).ipynb
│
├── outputs_baseline_cnn_curves (1).png
├── outputs_class_distribution.png
├── outputs_confusion_matrix.png
├── outputs_mobilenetv2_(fine-tuned)_curves.png
├── outputs_sample_images.png
├── outputs_sample_predictions.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MobileNetV2**
* **NumPy**
* **Pillow**
* **Streamlit**
* **Jupyter Notebook**

---

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/fruit-freshness-detection.git
```

```bash
cd fruit-freshness-detection
```

### 2. Create/activate your virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Results

The project evaluates both the baseline CNN and fine-tuned MobileNetV2 using visual performance metrics and prediction outputs.

The included visualizations provide insights into:

* Model learning behaviour
* Training vs validation performance
* Classification errors
* Class distribution
* Prediction examples

---

## 🔮 Future Improvements

Potential improvements include:

* Adding more fruit categories
* Increasing dataset size and diversity
* Further hyperparameter tuning
* Data augmentation
* Improving model interpretability
* Deploying the application online
* Adding confidence scores and detailed prediction reports

---



