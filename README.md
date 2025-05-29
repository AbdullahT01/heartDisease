# 🫀 Heart Disease Prediction Neural Network (From Scratch)

This project demonstrates a **neural network built entirely from scratch in Python** to predict whether a person has heart disease or not. It is based on clinical data from over 900 patients and does **not** rely on libraries like TensorFlow or PyTorch—everything from forward propagation to backpropagation is implemented manually.

## 🚀 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve patient outcomes. This project explores how a simple feedforward neural network can learn to identify patterns in patient data to predict the likelihood of heart disease.

## 📊 Dataset

The dataset includes information from 900+ patients and features such as:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol level  
- Fasting blood sugar  
- Resting ECG results  
- Maximum heart rate achieved  
- Exercise-induced angina  
- ST depression  
- Slope of the peak exercise ST segment  
- Number of major vessels colored by fluoroscopy  
- Thalassemia type  

> 🗂 *The dataset is preprocessed using normalization, one-hot encoding, and train-test splitting.*

## 🧠 Neural Network Architecture

- **Input Layer:** Number of neurons = number of features after preprocessing  
- **Hidden Layers:** Configurable; default uses 1–2 hidden layers  
- **Activation Functions:** ReLU for hidden layers, Sigmoid for output  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Gradient Descent (implemented from scratch)  
- **Output Layer:** Single neuron with Sigmoid activation for binary classification  

## 🛠 Features

- Pure Python implementation  
- Custom activation functions  
- Manual forward and backward propagation  
- Training and validation split  
- Accuracy evaluation  

## 📈 Sample Results

> Training Accuracy: ~88%  
> Test Accuracy: ~91%  
> (Add your actual numbers here once training is done)
