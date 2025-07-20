# 🏢Employee-Salary-Prediction-Using-Machine-Learning

A Machine Learning-powered Gradio web app that predicts whether an employee earns more than $50K per year based on demographic and professional features using different types of machine learning modules and techniques.

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

#### ➡️After Executing the command a localhost server at port 8080 will be created.
#### ➡️Open the Python notebook file(click twice to open).
![image alt](https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning/blob/main/Images/Jupyterlab.png?raw=true)

#### Run Each cell containing codes.

## 📊Some Sample Output 

### 1. Field Outputs
![image alt](https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning/blob/main/Images/sample_output.jpg?raw=true)

### 2. Scaling Outputs
#### i. MinMax Scaling
![image alt](https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning/blob/main/Images/MimMax%20Scaling.jpg?raw=true)

#### ii. Standard Scaling
![image alt](https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning/blob/main/Images/Standard%20Scaling.jpg?raw=true)

### 3. Best Model
![image alt](https://github.com/Codex-Amit/Employee-Salary-Prediction-Using-Machine-Learning/blob/main/Images/Best%20Model.jpg?raw=true)

#### Alongwith some more and exciting analysis methodologies.

## 📥 Dataset Requirements
For training or batch prediction, your CSV should contain these columns:
```text
age, workclass, education-num, marital-status, occupation, relationship,
race, gender, capital-gain, capital-loss, hours-per-week, native-country
```

---

## 🐛 Troubleshooting

#### Solutions to some common problems you might face

### ❗ 1. `ModuleNotFoundError: No module named 'gradio'`

**Cause**: Required Python packages not installed.

**Solution**:
#### Run the following command to install dependencies:
```yaml
pip install -r requirements.txt
```

### ❗ 2. `FileNotFoundError: [Errno 2] No such file or directory: 'adult.csv'`
**Cause**: Training file not found or not uploaded.

**Solution**:
Make sure to:

- Upload adult.csv in the Model Training tab.

- Provide your own CSV file in the correct format.

### ❗ 3. `ValueError: could not convert string to float`
 **Cause :** Incorrect column data types or unexpected missing values.
 
 **Solution:**
- Ensure your CSV file has valid numeric values for numeric columns.
- Avoid blank rows or unexpected strings.
- Use the provided adult.csv format as a template.

### ❗ 4. `Model not trained yet. Please upload training data first`
 **Cause:** Attempting prediction before training the model.

 **Solution:**

- First upload a training dataset and click on "Train Model".

- Then proceed to Single or Batch Prediction.

### ❗ 5. `App doesn't launch or crashes immediately`
**Solution:**

- Check Python version (recommended: Python 3.8+).

- Ensure all libraries are installed correctly.

- Run the script with `python salary_prediction_app.py` not `.ipynb`

---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This project was inspired by and made possible thanks to:

- 🧠 **Scikit-learn** for providing robust machine learning algorithms and utilities such as:
  - `GradientBoostingClassifier` used for salary classification
  - `LabelEncoder`, `StandardScaler`, and model evaluation functions
- 🐍 **Pandas** and **NumPy** for efficient data manipulation.
- 🌐 **Gradio** for making ML interfaces fast and user-friendly.
- 💡 Open-source contributors who help make ML tools accessible.
- 👨‍🏫 Community tutorials, blog posts, and notebooks that served as references.

Special thanks to all the developers, educators, and researchers who contribute to the open data and open source ecosystem ❤️.



## 📞 Support

- 📧 **Email**: amitkumarswain2005@gmail.com
---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**
