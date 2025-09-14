# Credit Card Customer Churn Prediction using ANN

This is a beginner deep learning project where I used an Artificial Neural Network (ANN) to predict whether a bank customer will churn (leave the bank) based on their demographic and account information.

## Project Overview

* Dataset: [Churn Modelling dataset](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction)
* Preprocessing:

  * Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
  * One-hot encoded categorical features (`Geography`, `Gender`)
  * Standardized numerical features using `StandardScaler`
* Model:

  * Dense layer (11 neurons, ReLU, input layer)
  * Dense layer (11 neurons, ReLU)
  * Output layer (1 neuron, Sigmoid for binary classification)

## Training

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy
* Epochs: 100
* Batch Size: 50
* Validation Split: 20%
* Early stopping used to prevent overfitting

## Results

* Test accuracy: **\~86%** (varies slightly per run)
* Visualized loss and accuracy curves for both training and validation sets

## Sample Prediction

The model outputs probabilities of churn for unseen test data, which are then converted to class labels (`0` = No churn, `1` = Churn) using a 0.5 threshold.

## Learnings

* How to preprocess tabular data with both categorical and numerical features
* Basics of building a binary classification ANN with TensorFlow/Keras
* Importance of scaling and proper activation functions in ANN models
* Model evaluation using accuracy, confusion matrix, and classification report
