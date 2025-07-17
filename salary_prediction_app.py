import gradio as gr
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        self.is_trained = False
        
    def preprocess_data(self, data):
        """Preprocess the input data"""
        processed_data = data.copy()
        
        # Handle missing values
        processed_data.replace('?', np.nan, inplace=True)
        
        # Fill missing values with mode for categorical columns
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object' and processed_data[col].isnull().sum() > 0:
                mode_val = processed_data[col].mode()[0] if len(processed_data[col].mode()) > 0 else 'Unknown'
                processed_data[col].fillna(mode_val, inplace=True)
        
        # Remove problematic entries
        if 'workclass' in processed_data.columns:
            processed_data = processed_data[~processed_data['workclass'].isin(['Never-worked', 'Without-pay'])]
        
        # Handle age outliers
        if 'age' in processed_data.columns:
            processed_data = processed_data[(processed_data['age'] >= 17) & (processed_data['age'] <= 75)]
        
        return processed_data
    
    def feature_engineering(self, data):
        """Apply feature engineering"""
        data = data.copy()
        
        # Handle education-num column name
        if 'education-num' in data.columns:
            data['educational-num'] = data['education-num']
            data.drop('education-num', axis=1, inplace=True)
        elif 'educational-num' not in data.columns:
            data['educational-num'] = 9  # Default to HS-grad equivalent
        
        # Create age groups
        data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 50, 65, 100], 
                                  labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])
        
        # Create hours category
        data['hours_category'] = pd.cut(data['hours-per-week'], bins=[0, 20, 40, 60, 100], 
                                       labels=['Part-time', 'Full-time', 'Overtime', 'Workaholic'])
        
        # Create capital features
        data['capital_net'] = data['capital-gain'] - data['capital-loss']
        data['has_capital_gain'] = (data['capital-gain'] > 0).astype(int)
        data['has_capital_loss'] = (data['capital-loss'] > 0).astype(int)
        
        # Education grouping
        education_mapping = {
            'Doctorate': 'Advanced', 'Prof-school': 'Advanced', 'Masters': 'Advanced',
            'Bachelors': 'Bachelors', 'Some-college': 'Some-college',
            'Assoc-acdm': 'Associate', 'Assoc-voc': 'Associate',
            'HS-grad': 'High-school', '12th': 'High-school', '11th': 'High-school',
            '10th': 'High-school', '9th': 'High-school', '7th-8th': 'Elementary'
        }
        
        if 'education' in data.columns:
            data['education_grouped'] = data['education'].map(education_mapping).fillna('High-school')
        else:
            data['education_grouped'] = 'High-school'
        
        # Drop unnecessary columns
        columns_to_drop = ['education', 'fnlwgt']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        
        return data
    
    def train_model(self, data):
        """Train the salary prediction model"""
        try:
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Feature engineering
            engineered_data = self.feature_engineering(processed_data)
            
            # Separate features and target
            X = engineered_data.drop('income', axis=1)
            y = engineered_data['income']
            
            # Initialize encoders
            self.label_encoders = {}
            self.target_encoder = LabelEncoder()
            
            # Encode categorical features
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
            
            # Encode target variable
            y_encoded = self.target_encoder.fit_transform(y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train model
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            return f"Model trained successfully! Accuracy: {accuracy:.4f}"
            
        except Exception as e:
            return f"Error training model: {str(e)}"
    
    def predict_single(self, age, workclass, education_num, marital_status, occupation, 
                      relationship, race, gender, capital_gain, capital_loss, 
                      hours_per_week, native_country):
        """Make prediction for a single instance"""
        
        if not self.is_trained:
            return "Model not trained yet. Please upload training data first."
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'educational-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'gender': [gender],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
            
            # Feature engineering
            input_data = self.feature_engineering(input_data)
            
            # Define training columns order
            training_columns = [
                'age', 'workclass', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'age_group', 'hours_category',
                'capital_net', 'has_capital_gain', 'has_capital_loss', 'education_grouped'
            ]
            
            # Reorder columns
            final_data = pd.DataFrame()
            for col in training_columns:
                if col in input_data.columns:
                    final_data[col] = input_data[col]
                else:
                    final_data[col] = 0
            
            # Encode categorical features
            for col, encoder in self.label_encoders.items():
                if col in final_data.columns:
                    try:
                        final_data[col] = encoder.transform(final_data[col].astype(str))
                    except:
                        final_data[col] = 0
            
            # Scale features
            scaled_data = self.scaler.transform(final_data)
            
            # Make prediction
            prediction = self.model.predict(scaled_data)[0]
            prediction_proba = self.model.predict_proba(scaled_data)[0]
            
            # Decode prediction
            result = self.target_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100
            
            return f"Predicted Salary: {result}\nConfidence: {confidence:.1f}%"
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def predict_batch(self, file):
        """Make predictions for batch data"""
        
        if not self.is_trained:
            return "Model not trained yet. Please upload training data first.", None
        
        try:
            # Read the uploaded file
            if file.name.endswith('.csv'):
                batch_data = pd.read_csv(file.name)
            else:
                return "Please upload a CSV file.", None
            
            # Store original data for output
            original_data = batch_data.copy()
            
            # Feature engineering
            batch_data = self.feature_engineering(batch_data)
            
            # Define training columns order
            training_columns = [
                'age', 'workclass', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'age_group', 'hours_category',
                'capital_net', 'has_capital_gain', 'has_capital_loss', 'education_grouped'
            ]
            
            # Reorder columns
            final_data = pd.DataFrame()
            for col in training_columns:
                if col in batch_data.columns:
                    final_data[col] = batch_data[col]
                else:
                    final_data[col] = 0
            
            # Encode categorical features
            for col, encoder in self.label_encoders.items():
                if col in final_data.columns:
                    try:
                        final_data[col] = encoder.transform(final_data[col].astype(str))
                    except:
                        final_data[col] = 0
            
            # Scale features
            scaled_data = self.scaler.transform(final_data)
            
            # Make predictions
            predictions = self.model.predict(scaled_data)
            predictions_proba = self.model.predict_proba(scaled_data)
            
            # Decode predictions
            decoded_predictions = self.target_encoder.inverse_transform(predictions)
            confidence_scores = np.max(predictions_proba, axis=1) * 100
            
            # Create results dataframe
            results = original_data.copy()
            results['Predicted_Salary'] = decoded_predictions
            results['Confidence_%'] = confidence_scores.round(1)
            
            return f"Batch prediction completed for {len(results)} records.", results
            
        except Exception as e:
            return f"Error in batch prediction: {str(e)}", None

# Initialize predictor
predictor = SalaryPredictor()

# Define input options
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
marital_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
gender_options = ['Female', 'Male']
country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

def train_model_interface(file):
    """Interface function to train the model"""
    if file is None:
        return "Please upload a CSV file to train the model."
    
    try:
        # Read the uploaded file
        if file.name.endswith('.csv'):
            data = pd.read_csv(file.name)
        else:
            return "Please upload a CSV file."
        
        # Train the model
        result = predictor.train_model(data)
        return result
        
    except Exception as e:
        return f"Error loading training data: {str(e)}"

def predict_interface(age, workclass, education_num, marital_status, occupation, 
                     relationship, race, gender, capital_gain, capital_loss, 
                     hours_per_week, native_country):
    """Interface function for single prediction"""
    return predictor.predict_single(age, workclass, education_num, marital_status, 
                                   occupation, relationship, race, gender, 
                                   capital_gain, capital_loss, hours_per_week, 
                                   native_country)

def batch_predict_interface(file):
    """Interface function for batch prediction"""
    if file is None:
        return "Please upload a CSV file for batch prediction.", None
    
    message, results = predictor.predict_batch(file)
    return message, results

# Create Gradio interface
with gr.Blocks(title="Employee Salary Prediction", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🏢 Employee Salary Prediction System Using Machine Learning 
    
    This application predicts whether an employee's salary is above or below $50K based on demographic and work-related features.
    
    ## Instructions:
    1. **First**, upload a training dataset (CSV file) to train the model
    2. **Then**, use either single prediction or batch prediction
    """)
    
    with gr.Tab("📊 Model Training"):
        gr.Markdown("### Upload Training Data")
        gr.Markdown("Upload a CSV file containing employee data with an 'income' column for training.")
        
        with gr.Row():
            train_file = gr.File(label="Upload Training CSV", file_types=[".csv"])
            train_button = gr.Button("Train Model", variant="primary")
        
        train_output = gr.Textbox(label="Training Status", lines=3)
        
        train_button.click(
            fn=train_model_interface,
            inputs=[train_file],
            outputs=[train_output]
        )
    
    with gr.Tab("👤 Single Prediction"):
        gr.Markdown("### Enter Employee Details")
        
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=30, minimum=17, maximum=100)
                workclass = gr.Dropdown(label="Work Class", choices=workclass_options, value="Private")
                education_num = gr.Number(label="Education Number", value=9, minimum=1, maximum=16)
                marital_status = gr.Dropdown(label="Marital Status", choices=marital_options, value="Never-married")
                occupation = gr.Dropdown(label="Occupation", choices=occupation_options, value="Tech-support")
                relationship = gr.Dropdown(label="Relationship", choices=relationship_options, value="Not-in-family")
            
            with gr.Column():
                race = gr.Dropdown(label="Race", choices=race_options, value="White")
                gender = gr.Dropdown(label="Gender", choices=gender_options, value="Male")
                capital_gain = gr.Number(label="Capital Gain", value=0, minimum=0)
                capital_loss = gr.Number(label="Capital Loss", value=0, minimum=0)
                hours_per_week = gr.Number(label="Hours per Week", value=40, minimum=1, maximum=100)
                native_country = gr.Dropdown(label="Native Country", choices=country_options, value="United-States")
        
        predict_button = gr.Button("Predict Salary", variant="primary")
        single_output = gr.Textbox(label="Prediction Result", lines=3)
        
        predict_button.click(
            fn=predict_interface,
            inputs=[age, workclass, education_num, marital_status, occupation, 
                   relationship, race, gender, capital_gain, capital_loss, 
                   hours_per_week, native_country],
            outputs=[single_output]
        )
    
    with gr.Tab("📋 Batch Prediction"):
        gr.Markdown("### Upload Data for Batch Prediction")
        gr.Markdown("Upload a CSV file containing employee data (without 'income' column) for batch prediction.")
        
        with gr.Row():
            batch_file = gr.File(label="Upload Batch CSV", file_types=[".csv"])
            batch_button = gr.Button("Predict Batch", variant="primary")
        
        batch_output = gr.Textbox(label="Batch Prediction Status", lines=2)
        batch_results = gr.Dataframe(label="Prediction Results", wrap=True)
        
        batch_button.click(
            fn=batch_predict_interface,
            inputs=[batch_file],
            outputs=[batch_output, batch_results]
        )
    
    with gr.Tab("📖 About"):
        gr.Markdown("""
        ## About This Application
        
        This Employee Salary Prediction system uses machine learning to predict whether an employee's salary exceeds $50K annually.
        
        ### Features Used:
        - **Demographic**: Age, Race, Gender, Native Country
        - **Work Related**: Work Class, Occupation, Hours per Week
        - **Personal**: Marital Status, Relationship, Education Level
        - **Financial**: Capital Gain, Capital Loss
        
        ### Model Features:
        - **Algorithm**: Gradient Boosting Classifier
        - **Feature Engineering**: Age groups, work hour categories, capital features
        - **Preprocessing**: Missing value handling, outlier detection
        - **Scaling**: Standard normalization
        
        ### File Format Requirements:
        - **Training Data**: Must include 'income' column with values '<=50K' or '>50K'
        - **Batch Prediction**: Should not include 'income' column
        - **Format**: CSV files only
        
        ### Column Names Expected:
        `age`, `workclass`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`
        """)

# Launch the app
if __name__ == "__main__":
    app.launch(debug=True, share=True)