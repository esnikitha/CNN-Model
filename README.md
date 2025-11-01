# 🧠 CNN Model on MNIST Dataset – Handwritten Digit Classification

## 📘 Project Overview

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the **MNIST dataset**. The model automatically learns visual patterns and features, achieving high accuracy in recognizing digits from grayscale images.

## 🎯 Objective

To develop an end-to-end deep learning model capable of classifying handwritten digits using CNNs, showcasing core concepts of computer vision and neural network design.

## 🧩 Dataset

**MNIST Dataset** — a collection of 70,000 grayscale images of handwritten digits (28x28 pixels), split into:

* 60,000 training images
* 10,000 testing images

Each image is labeled with the correct digit (0–9).

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy & Matplotlib

## 🏗️ Model Architecture

1. **Conv2D (32 filters, 3×3 kernel, ReLU)**
2. **MaxPooling2D (2×2)**
3. **Conv2D (64 filters, 3×3 kernel, ReLU)**
4. **MaxPooling2D (2×2)**
5. **Flatten Layer**
6. **Dense (128 neurons, ReLU)**
7. **Dropout (0.5)**
8. **Dense (10 neurons, Softmax)**

## 🚀 Training Details

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 128
* **Epochs:** 10

## 📈 Results

* **Test Accuracy:** ~99%
* **Loss:** < 0.04
  The CNN model successfully identifies handwritten digits with near-perfect accuracy.

## 🖼️ Visualization

* Training vs Validation Accuracy & Loss
* Sample Predictions with Actual Labels

## 💾 Model Output

The trained model is saved as:

```
mnist_cnn_model.h5
```

## 🧾 Key Learnings

* Understanding CNN architecture for image recognition
* Preprocessing grayscale image data
* Visualizing training performance and predictions
* Applying dropout to reduce overfitting

## 🌟 Future Improvements

* Implement data augmentation to enhance robustness
* Experiment with deeper CNNs or transfer learning models
* Deploy the model using a web app (Streamlit or Flask)

---

**GitHub Tags:** `#DeepLearning` `#ComputerVision` `#TensorFlow` `#Keras` `#MNIST` `#CNN` `#MachineLearning`

---

📍 *A simple yet powerful example of how CNNs can be used for digit recognition — a foundational project for anyone starting in Deep Learning.*
