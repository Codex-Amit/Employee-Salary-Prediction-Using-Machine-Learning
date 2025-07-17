# 🏢Employee-Salary-Prediction-Using-Machine-Learning

A Machine Learning-powered Gradio web app that predicts whether an employee earns more than $50K per year based on demographic and professional features.

---

## 🚀 Features

- ✅ Upload custom dataset and train the model.
- ✅ Predict salary for a single employee via form input.
- ✅ Batch prediction via CSV file upload.
- ✅ Clean preprocessing, outlier handling, and feature engineering.
- ✅ Uses `GradientBoostingClassifier` with scaling and encoding.

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

---
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
## 📥 Dataset Requirements
For training or batch prediction, your CSV should contain these columns:
```text
age, workclass, education-num, marital-status, occupation, relationship,
race, gender, capital-gain, capital-loss, hours-per-week, native-country
```
Training Data: Must include income column with values <=50K or >50K

Batch Prediction Data: Should not include income column

---
## 🛠 Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Gradio for UI
- Gradient Boosting Classifier
