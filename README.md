# **Titanic Machine Learning Project**

This project aims to create a machine learning model to predict the survival status of Titanic passengers based on various features available in the dataset.

Dataset source : [Kaggle](https://www.kaggle.com/competitions/titanic)

## **1. Project Goal**

The main objective of this project is to predict the survival status (Survived column) of Titanic's passengers using given features from the training data.

## **2. Data Understanding**

We use visual and statistical analysis to gain a deeper understanding of the data:

Bar charts are used for visual analysis to infer information about survival status of passengers from categorical columns such as the embarkation port.
The survival percentage is calculated based on each categorical column.
A correlation heatmap is created to show the correlation between each numerical feature and the Survived column.

## **3. Data Preparation**

This step includes:

Checking for missing values and handling them accordingly.
Changing the data types of several columns for compatibility with the machine learning models.

## **4. Feature Engineering**

In this step, new features are created that can aid in the modelling process. Some of these include 'IsAlone' and 'HasCabin'.

## **5. Modelling**

This is the core part where we train different machine learning models and select the best performing one based on various metric evaluations, including accuracy, precision, recall, F1 score, and ROC AUC score.

## **6. Hyperparameter Tuning**

In this stage, the selected model (LightGBM) is tuned for better performance and efficiency.

## **7. Test Model**

Finally, we test our model using unseen data and evaluate its performance.

## **Prerequisites**

The following Python libraries are required to run this project:

1. Standard Libraries:
- Numpy
- Pandas
- Pickle

2. Visualization:
- Matplotlib
- Seaborn

3. Preprocessing, Model Selection, and Machine Learning Classifiers:
- Scipy
- Scikit-learn
- XGBoost
- LightGBM
- Category_encoders
- Imbalanced-learn