# ❤️ Heart Disease Prediction System

> **Binary Classification · Logistic Regression · Clinical Data · Model Deployment**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-82.76%25-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

Built an **end-to-end binary classification pipeline** to predict heart disease risk from structured patient clinical data. The project covers the full ML workflow — EDA, stratified train/test splitting, model training, evaluation with clinical priority metrics, automated report generation, and model serialization for deployment.

**Core objective:** Predict whether a patient has heart disease (output = 1) or not (output = 0), with emphasis on **maximizing recall** to minimize clinically costly false negatives (missed diagnoses).

---

## 📊 Model Performance

| Metric | Class 0 (No Disease) | Class 1 (Disease) | Overall |
|--------|---------------------|-------------------|---------|
| Precision | 0.86 | 0.81 | 0.83 |
| Recall | 0.72 | **0.91** | 0.83 |
| F1-Score | 0.78 | 0.86 | 0.83 |
| Support | 25 | 33 | 58 |

**Overall Accuracy: 82.76%**

> The model achieves **91% recall on heart disease cases** — meaning it correctly identifies 91% of actual patients with heart disease. This is the clinically critical metric: a false negative (telling a sick patient they are healthy) is far more dangerous than a false positive.

---

## 📁 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Heart Disease Dataset (`heart.csv`) |
| Total samples | 289 |
| Features | 13 clinical attributes |
| Target | `output` — 0 (no disease), 1 (disease) |
| Missing values | None |
| Class distribution | 165 positive (57%), 124 negative (43%) |

**Key clinical features used:**

| Feature | Description |
|---------|-------------|
| `age` | Patient age |
| `sex` | Biological sex |
| `cp` | Chest pain type (4 categories) |
| `trtbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `thalachh` | Maximum heart rate achieved |
| `exng` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `caa` | Number of major vessels (0–3) |
| `thall` | Thalassemia type |

---

## 📈 Visualizations

### Target Distribution
![Target Distribution](https://github.com/Dushdeva/Heart_disease_prediction_ml/blob/main/Report/target_distribution.png?raw=true)

### Feature Correlation Heatmap
![Correlation Heatmap](https://github.com/Dushdeva/Heart_disease_prediction_ml/blob/main/Report/correlation_heatmap.png?raw=true)

### Confusion Matrix
![Confusion Matrix](https://github.com/Dushdeva/Heart_disease_prediction_ml/blob/main/Report/confusion_matrix.png?raw=true)

---

## 🏗️ Pipeline Architecture

```
heart.csv (289 rows, 13 features)
        │
        ▼
  EDA & Validation
  (shape check, missing values, target distribution, correlation heatmap)
        │
        ▼
  Stratified Train/Test Split
  (80/20 → 231 train / 58 test, random_state=99)
        │
        ▼
  Logistic Regression (lbfgs, max_iter=1000)
        │
        ▼
  Evaluation
  (accuracy, precision, recall, F1, confusion matrix)
        │
        ▼
  ┌─────────────────────────────────┐
  │  report.txt (classification     │
  │  report + accuracy summary)     │
  │  heart_disease_model.pkl        │
  │  (joblib serialized model)      │
  └─────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Heart_disease_prediction_ml/
│
├── heart_disease_prediction.ipynb    # Main notebook (Colab-ready)
├── requirements.txt
├── README.md
│
├── dataset/
│   └── heart.csv                     # Input data
│
└── Report/                           # Auto-generated on run
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── report.txt                    # Accuracy + classification report
    └── heart_disease_model.pkl       # Serialized trained model
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open the notebook in Colab
2. Upload `dataset/heart.csv` to the Colab session
3. `Runtime → Run all`
4. All outputs auto-saved to `Report/`

### Option 2: Local (Python 3.10+)

```bash
git clone https://github.com/Dushdeva/Heart_disease_prediction_ml.git
cd Heart_disease_prediction_ml
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

### Option 3: Load saved model directly

```python
import joblib
import pandas as pd

model = joblib.load("Report/heart_disease_model.pkl")

# Example patient data (13 features)
patient = pd.DataFrame([{
    "age": 52, "sex": 1, "cp": 0, "trtbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalachh": 168,
    "exng": 0, "oldpeak": 1.0, "slp": 2, "caa": 2, "thall": 3
}])

prediction = model.predict(patient)
print("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")
```

---

## 🔮 Identified Improvements

- [ ] **Add StandardScaler** — feature scaling can improve Logistic Regression convergence and coefficient interpretability
- [ ] **k-Fold Cross Validation** — 5 or 10-fold CV for more robust accuracy estimate than single 80/20 split
- [ ] **Try ensemble models** — Random Forest / XGBoost for likely accuracy improvement (target: 88%+)
- [ ] **Deploy via Flask/Streamlit** — expose `/predict` endpoint accepting patient JSON, returning risk probability
- [ ] **SHAP values** — feature importance explanation for clinical interpretability

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML model | `scikit-learn` (Logistic Regression) |
| Model persistence | `joblib` |
| Environment | Google Colab / Jupyter |

---

*Built by [Devang Yadav](https://github.com/Dushdeva) — B.Tech CSE (AI), SKIT Jaipur*
