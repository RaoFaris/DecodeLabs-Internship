# Project 2 – Data Classification Using AI

## Overview

This project was developed as part of the DecodeLabs AI Internship. The goal was to build a simple machine learning classification model using the Iris dataset.

The project demonstrates the basic workflow of a supervised learning problem, including loading a dataset, exploring the data, splitting it into training and testing sets, training a classification model, and evaluating its performance.

## Objectives

The objectives of this project are:

- Load and understand a dataset.
- Explore the dataset before training.
- Split the data into training and testing sets.
- Train a classification model.
- Evaluate the model using standard performance metrics.
- Make predictions on new, unseen data.

## Technologies Used

- Python 3
- NumPy
- Pandas
- scikit-learn

## Dataset

This project uses the **Iris dataset**, which is included with scikit-learn and does not require any external download.

Dataset details:

- 150 samples
- 4 input features
- 3 flower species:
  - Setosa
  - Versicolor
  - Virginica

The four features used for classification are:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Classification Algorithm

The model uses the **K-Nearest Neighbors (KNN)** algorithm with **K = 5**.

Since KNN is a distance-based algorithm, the feature values are standardized using **StandardScaler** before training the model.

## Project Structure

```
Project-2-Data-Classification/
│
├── main.py
└── README.md
```

## How to Run

1. Install Python 3.
2. Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

3. Run the program:

```bash
python main.py
```

## Example Output

The program displays:

- Dataset information
- Training and testing split
- Model training status
- Accuracy score
- Confusion matrix
- Classification report
- Predictions for new sample data

Example:

```text
Accuracy: 100.00%

Confusion Matrix:
[[10 0 0]
 [0 9 0]
 [0 0 11]]

Predicted species:
Setosa
Versicolor
Virginica
```

## Concepts Used

This project demonstrates:

- Supervised Machine Learning
- Classification
- Dataset exploration
- Feature scaling
- Train/test splitting
- Model training
- Model evaluation
- Predictions on unseen data

## What I Learned

Through this project, I learned how to:

- Load and explore datasets using scikit-learn and Pandas.
- Prepare data before training a machine learning model.
- Split data into training and testing sets.
- Train a K-Nearest Neighbors classifier.
- Evaluate a model using accuracy, confusion matrix, and classification report.
- Make predictions using trained models.

## Possible Improvements

Some improvements that could be added in the future include:

- Comparing multiple classification algorithms.
- Visualizing the dataset using graphs.
- Performing hyperparameter tuning.
- Testing the model on different datasets.
- Building a simple graphical interface for predictions.
