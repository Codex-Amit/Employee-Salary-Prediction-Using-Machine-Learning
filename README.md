# 🏢Employee-Salary-Prediction-Using-Machine-Learning

A Machine Learning-powered Gradio web app that predicts whether an employee earns more than $50K per year based on demographic and professional features usong different types of machine learning modules and techniques.

---

## 🚀 Features

- ✅ Upload custom dataset and train the model.
- ✅ Predict salary for a single employee via form input.
- ✅ Batch prediction via CSV file upload.
- ✅ Clean preprocessing, outlier handling, and feature engineering.
- ✅ Uses `GradientBoostingClassifier` with scaling and encoding.

---

## 🤖 ML Models Implemented

The following machine learning algorithms were tested:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest Classifier
- Gradient Boosting Classifier

Evaluation metrics:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 🧠 Deep Learning Approach

A feed-forward Neural Network was built using **TensorFlow/Keras**:

- Layers: Dense layers with ReLU activation
- Optimizer: Adam
- Metrics: Accuracy
- Additional techniques: Dropout, Batch Normalization

---

## 📁 Project Structure
```yaml
Employee-Salary-Prediction-Using-Machine-Learning/
│
├── salary_prediction_app.py     # Main Gradio application
├── _Employee_Salary_Prediction.ipynb     # Jupyter notebook version
├── adult.csv     # Sample dataset (UCI Adult Dataset)
├── requirements.txt     # List of dependencies
└── README.md     # Project documentation
```
#### # Note that the python version should be of version 3.12.0. As tensorflow is used in the project which supports only version upto 3.12.0. And using above it may cause error or may don't give the accurate result.


---

## ⚙️ Installation Guide

### 1. Clone the Repository

```yaml
git clone https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning.git
cd Employee-Salary-Prediction-Using-Machine-Learning
```
### 2. (Optional) Create Virtual Environment

```yaml
For creating a virtual environment:
python -m venv venv

Activating venv On Windows:
venv\Scripts\activate

Activating venv On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```yaml
pip install -r requirements.txt
```

### 4. Run the App
```yaml
python salary_prediction_app.py
```
 Gradio will launch in your browser at http://localhost:7860

## 📊 Sample Outputs

### ✅ Model Training
```yaml
Model trained successfully! Accuracy: 0.8732
```

### ✅ Single Prediction
```yaml
Predicted Salary: >50K
Confidence: 89.7%
```

### ✅ Batch Prediction

| Age | Workclass | Education | ... | Predicted\_Salary | Confidence\_% |
| --- | --------- | --------- | --- | ----------------- | ------------- |
| 39  | Private   | Bachelors | ... | >50K              | 90.2          |
| 28  | Private   | HS-grad   | ... | <=50K             | 78.6          |

---

## For more detailed Analysis ( reffer to the Jupyter Notebook)

## 1. Jupyter Lab Installation
```yaml
cd Employee-Salary-Prediction-Using-Machine-Learning
python -m install jupyterlab
```

## 2. Run The Notebook
```yaml
python -m jupyterlab
```
![image alt](https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning/blob/main/Images/cmd.jpg?raw=true)
---

## 📥 Dataset Requirements
For training or batch prediction, your CSV should contain these columns:
```text
age, workclass, education-num, marital-status, occupation, relationship,
race, gender, capital-gain, capital-loss, hours-per-week, native-country
```
---
## 🛠 Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Gradio for UI
- Gradient Boosting Classifier
