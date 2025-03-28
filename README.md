# Deep-Learning-Challenge

# Overview
This project involves using machine learning to predict the success of Alphabet Soup-funded organizations based on various features. The goal is to preprocess the data, build and evaluate a deep neural network model for binary classification, and optimize the model to achieve higher than 75% accuracy.

# Steps:
Step 1: Preprocess the Data
  - Loaded and preprocessed the charity_data.csv dataset.

  - Dropped irrelevant columns (EIN, NAME).

  - Replaced rare categorical variables in the APPLICATION_TYPE and CLASSIFICATION columns with the label "Other".

  - Encoded categorical variables using pd.get_dummies().

  - Split the data into features (X) and target (y), and then into training and testing datasets.

  - Scaled the data using StandardScaler().

Step 2: Build and Train the Model
  - Built a deep neural network with two hidden layers (128 and 64 neurons) and a sigmoid output layer for binary classification.

  - The model was compiled using binary_crossentropy as the loss function and adam as the optimizer.

  - Trained the model for 100 epochs with a batch size of 64.

Step 3: Evaluate and Save the Model
  - Evaluated the model's performance on the test data to measure loss and accuracy.

  - Saved the trained model as AlphabetSoupCharity.h5.

Step 4: Model Optimization (Optional)
  - Experimented with model optimization methods to improve accuracy, including adjustments to layers, neurons, and activation functions.

# Requirements
Python 3.x

Pandas

Scikit-learn

TensorFlow (version 2.x)

# Output
The model's loss and accuracy on the test data.

A saved model file AlphabetSoupCharity.h5.
