# Diabetes Health Indicators

This repository contains a machine learning model that predicts whether a patient has diabetes or not, based on various health indicators. The model is built using Python and uses the Random Forest algorithm for classification. The model has been trained on the Diabetes Health Indicators Dataset available on Kaggle.

## Dataset
The dataset can be accessed at https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset. It contains information about various health indicators such as glucose level, blood pressure, body mass index, etc. for a group of patients, along with their diabetes status.

## Data Cleaning
The dataset has been cleaned and pre-processed to remove any duplicates and null values. The class imbalance has also been checked and addressed by using stratified sampling while splitting the data into training and testing sets.

## Feature Selection
The feature selection process involves selecting the most important features that contribute to the prediction of diabetes. Three methods have been used for feature selection: Mutual Information, Chi-Squared, and Pearson Correlation.

## Models
The model has been built using five different classification algorithms: Logistic Regression, Random Forest Classifier, Decision Tree Classifier, K Neighbors Classifier, and Support Vector Machines. The performance of each model has been evaluated using various metrics such as AUC, accuracy, precision, recall, and F1 score.

## Results
After evaluating the performance of each model, the Random Forest Classifier with 10 features selected using Pearson Correlation wasfound to be the best performing model. It achieved an AUC of 0.94, accuracy of 0.88, precision of 0.94, recall of 0.81, and an F1 score of 0.87.

## Model Saving
The model has been saved in the ONNX format, which is a widely used open standard for representing machine learning models. This makes it easy to deploy the model in various platforms and programming languages.

## Usage
To use the model, simply load the saved ONNX file and feed it with the relevant input data. The model will then output a prediction of whether the patient has diabetes or not.

``` python
import numpy
import onnxruntime as rt

sess = rt.InferenceSession("diabetes-model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

pred = sess.run(
    [output_name], {input_name:X_test.values.astype(numpy.float32)})[0]
```
## Conclusion
This project demonstrates how machine learning can be used to predict the likelihood of a patient having diabetes based on various health indicators. The best performing model achieved an AUC of 0.94, which indicates that it has a high level of accuracy in predicting diabetes. The model can be further improved by incorporating additional features and using more advanced machine learning techniques.
