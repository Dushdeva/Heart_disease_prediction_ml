```Markdown
# Heart Disease Prediction System

## Overview
This project is a Python-based machine learning pipeline that analyzes patient health data to predict the likelihood of heart disease. Using a Logistic Regression framework, the system processes patient metrics, evaluates model performance, and automatically exports visual and textual diagnostic reports.

## Features
* **Automated Data Pipeline:** Loads and validates data shapes, checking for missing values automatically.
* **Exploratory Data Analysis (EDA):** Generates and saves target distribution plots and feature correlation heatmaps.
* **Reproducible Splits:** Utilizes stratified sampling ($80/20$ split) with a fixed random state to ensure consistent training evaluation.
* **Automated Reporting:** Generates a comprehensive evaluation summary (`report.txt`) alongside visual confusion matrices.
* **Model Serialization:** Saves the trained Logistic Regression model as a `.pkl` file for future deployment.

## Technologies Used
* **Language:** Python
* **Data Libraries:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **Model Persistence:** Joblib

## Dataset Structure
The system processes `dataset/heart.csv`, utilizing key medical features including:
* Age & Biological Sex
* Chest Pain Type
* Resting Blood Pressure & Cholesterol Levels
* Fasting Blood Sugar
* Maximum Heart Rate Achieved

## Project Directory Layout
Once executed, the project establishes the following structure:
```text
├── dataset/
│   └── heart.csv                 # Required input data
├── heart_disease_prediction.ipynb # Core execution notebook
└── Report/                       # Generated automatically on runtime
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── report.txt                # Plain-text evaluation summary
    └── heart_disease_model.pkl   # Saved Logistic Regression model
```
## Learning Outcomes

I learned a lot from working on the Heart Disease Prediction System. I got experience with:

* Programming, in Python

* Analyzing data

* Visualizing data

* The basics of machine learning

* How to check if a model is working well

## Future Improvements

* I want to use Flask or Streamlit to make the model available online.

* I want to add a user- interface to the model.

* I want to make the model better at predicting heart disease by using advanced algorithms.

## Author

Devang Yadav
