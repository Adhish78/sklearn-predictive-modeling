# Predictive Modeling of Visceral Fat Levels for Health Risk Stratification

## Project Overview
This project aims to develop a predictive modeling system to estimate visceral fat levels for health risk stratification. Using various machine learning techniques, the system predicts whether an intervention is required based on a set of health-related features.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Results](#results)

## Introduction
The project involves the following steps:
1. Data exploration and cleaning.
2. Data transformation and scaling.
3. Building and evaluating multiple machine learning models.
4. Hyperparameter tuning for the best model performance.

## Dataset
The dataset consists of various health-related features for male subjects, including age, BMI, height, weight, waist circumference, blood pressure, daily activity duration, and smoking habits. The target variable indicates whether an intervention is required based on the visceral fat volume.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Data Preprocessing
The preprocessing steps include:
1. Loading the dataset.
2. Handling missing values by filling or removing them.
3. Removing outliers based on domain-specific thresholds.
4. Dropping irrelevant columns.
5. Transforming and scaling features.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) includes:
- Statistical summary of the dataset.
- Visualization of data distributions and relationships.
- Analysis of the target variable distribution.

## Model Building and Evaluation
The project implements and evaluates several machine learning models:
1. **Naïve Bayes Classifier**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors (k-NN) Classifier**
4. **Artificial Neural Network (MultiLayer Perceptron)**

Each model is trained, tested, and evaluated using classification metrics such as accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.

### Hyperparameter Tuning
Hyperparameter tuning is performed using GridSearchCV to optimize the performance of the Artificial Neural Network model.

## Results
The best model, determined through hyperparameter tuning, achieves high accuracy in predicting health risk categories. The results are visualized using classification reports, confusion matrices, and ROC curves.
