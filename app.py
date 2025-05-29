from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

# Database setup
DATABASE = 'predictions.db'

def init_db():
    """Initialize the database with predictions table"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Create table with all form fields as columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                employee_name TEXT NOT NULL,
                gender TEXT NOT NULL,
                age INTEGER NOT NULL,
                marital_status TEXT NOT NULL,
                education TEXT NOT NULL,
                position TEXT NOT NULL,
                years_of_service INTEGER NOT NULL,
                annual_salary REAL NOT NULL,
                recruitment_source TEXT NOT NULL,
                engagement_score INTEGER NOT NULL,
                satisfaction_score INTEGER NOT NULL,
                special_projects INTEGER NOT NULL,
                training_hours INTEGER NOT NULL,
                absences INTEGER NOT NULL,
                days_late INTEGER NOT NULL,
                predicted_performance TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Also create the history table (simplified version for history page display)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                employee_name TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                predicted_performance TEXT NOT NULL,
                position TEXT NOT NULL,
                prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")

def save_prediction(form_data, predicted_performance):
    """Save a prediction with all form data to the database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Insert into predictions table (full data)
        cursor.execute('''
            INSERT INTO predictions (
                employee_id, employee_name, gender, age, marital_status, education,
                position, years_of_service, annual_salary, recruitment_source,
                engagement_score, satisfaction_score, special_projects, training_hours,
                absences, days_late, predicted_performance, prediction_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            form_data['employee_id'],
            form_data['employee_name'],
            form_data['gender'],
            int(form_data['age']),
            form_data['marital_status'],
            form_data['education'],
            form_data['position'],
            int(form_data['years_of_service']),
            float(form_data['annual_salary']),
            form_data['recruitment_source'],
            int(form_data['engagement_score']),
            int(form_data['satisfaction_score']),
            int(form_data['special_projects']),
            int(form_data['training_hours']),
            int(form_data['absences']),
            int(form_data['days_late']),
            predicted_performance,
            prediction_date
        ))
        
        # Also insert into history table (for history page display)
        cursor.execute('''
            INSERT INTO history (employee_id, employee_name, prediction_date, predicted_performance, position)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            form_data['employee_id'],
            form_data['employee_name'],
            prediction_date,
            predicted_performance,
            form_data['position']
        ))
        
        conn.commit()
        conn.close()
        print("Prediction saved successfully!")
        
    except Exception as e:
        print(f"Error saving prediction: {e}")

def get_all_predictions():
    """Retrieve all predictions from history table for display"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, employee_name, prediction_date, predicted_performance, position
            FROM history
            ORDER BY prediction_timestamp DESC
        ''')
        predictions = cursor.fetchall()
        conn.close()
        return predictions
    except Exception as e:
        print(f"Error retrieving predictions: {e}")
        return []

def get_full_predictions_data():
    """Retrieve all complete prediction data from predictions table"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, employee_name, gender, age, marital_status, education,
                   position, years_of_service, annual_salary, recruitment_source,
                   engagement_score, satisfaction_score, special_projects, training_hours,
                   absences, days_late, predicted_performance, prediction_date
            FROM predictions
            ORDER BY prediction_timestamp DESC
        ''')
        predictions = cursor.fetchall()
        conn.close()
        return predictions
    except Exception as e:
        print(f"Error retrieving full predictions: {e}")
        return []

def clear_all_predictions():
    """Clear all predictions from both tables"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        cursor.execute('DELETE FROM history')
        conn.commit()
        conn.close()
        print("All predictions cleared successfully!")
    except Exception as e:
        print(f"Error clearing predictions: {e}")

# Mapping dictionaries
EDUCATION_MAPPING = {
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'PhD': 4
}

PERFORMANCE_LABELS = {
    0: "Poor Performance",
    1: "Needs Improvements", 
    2: "Meets Expectations",
    3: "Exceeds Expectations"
}

# Dummy prediction function (since model files might not exist)
def dummy_predict(form_data):
    """Create a dummy prediction based on simple rules"""
    # Simple scoring based on form data
    score = 0
    
    # Education score (0-4)
    score += EDUCATION_MAPPING.get(form_data['education'], 2)
    
    # Engagement and satisfaction scores (normalize to 0-4)
    score += (int(form_data['engagement_score']) - 1)
    score += (int(form_data['satisfaction_score']) - 1)
    
    # Positive factors
    score += min(int(form_data['special_projects']) / 2, 2)
    score += min(int(form_data['training_hours']) / 20, 2)
    
    # Negative factors
    score -= min(int(form_data['absences']) / 3, 2)
    score -= min(int(form_data['days_late']) / 2, 2)
    
    # Normalize score to 0-3 range
    score = max(0, min(score / 4, 3))
    
    return int(score)

# Load model and scaler (with fallback to dummy prediction)
model = None
scaler = None

try:
    if os.path.exists('tuned_random_forest_model.pkl') and os.path.exists('scaler.pkl'):
        with open('tuned_random_forest_model.pkl', 'rb') as f:
            model = joblib.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = joblib.load(f)
        
        print("Model and scaler loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Scaler type: {type(scaler)}")
        
        if hasattr(model, 'predict'):
            print("Model has predict method ✓")
        else:
            print("ERROR: Model does not have predict method!")
            model = None
            scaler = None
    else:
        print("Model files not found. Using dummy prediction function.")
        
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print("Using dummy prediction function.")
    model = None
    scaler = None

def preprocess_input(form_data):
    """Preprocess form data to match model training format"""
    try:
        # Create DataFrame with exact column names as in training
        df = pd.DataFrame([{
            'Education': form_data['education'],
            'Salary': float(form_data['annual_salary']),
            'EmpSatisfaction_Score': float(form_data['satisfaction_score']),
            'EmpEngagement_Score': float(form_data['engagement_score']),
            'SpecialProjects_Handled': float(form_data['special_projects']),
            'TrainingHours': float(form_data['training_hours']),
            'DaysLateLast30': float(form_data['days_late']),
            'Absences': float(form_data['absences']),
            'Age': float(form_data['age']),
            'Tenure_Years': float(form_data['years_of_service']),
            'Gender': form_data['gender'],
            'MaritalDesc': form_data['marital_status'],
            'Position': form_data['position'],
            'RecruitmentSource': form_data['recruitment_source']
        }])
        
        # Map education to numeric values
        df['Education'] = df['Education'].map(EDUCATION_MAPPING)
        
        # One-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=['Gender', 'MaritalDesc', 'Position', 'RecruitmentSource'], drop_first=True)
        
        # Define all expected features in the exact order the model was trained on
        expected_features = [
            'Education', 'Salary', 'EmpSatisfaction_Score', 'EmpEngagement_Score',
            'SpecialProjects_Handled', 'TrainingHours', 'DaysLateLast30', 'Absences',
            'Age', 'Tenure_Years', 'Gender_Male', 'MaritalDesc_Married', 'MaritalDesc_Single',
            'Position_Business Analyst', 'Position_Cybersecurity Analyst', 'Position_Data Analyst',
            'Position_Database Administrator', 'Position_Digital Marketing Specialist',
            'Position_Engineering Manager', 'Position_Finance Manager', 'Position_HR Manager',
            'Position_HR Specialist', 'Position_IT Manager', 'Position_IT Specialist',
            'Position_Market Research Analyst', 'Position_Marketing Coordinator',
            'Position_Marketing Manager', 'Position_Network Engineer', 'Position_Production Manager',
            'Position_Production Technician', 'Position_QA Engineer', 'Position_Sales Manager',
            'Position_Sales Representative', 'Position_Software Engineer',
            'Position_Technical Support Specialist', 'RecruitmentSource_Google Search',
            'RecruitmentSource_Indeed', 'RecruitmentSource_Job Fair', 'RecruitmentSource_Other',
            'RecruitmentSource_Website'
        ]
        
        # Add missing columns with 0 values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training order
        df = df[expected_features]
        
        # Define numeric features to scale (same as in training)
        numeric_features = [
            'Salary', 'EmpEngagement_Score', 'EmpSatisfaction_Score',
            'SpecialProjects_Handled', 'DaysLateLast30', 'TrainingHours',
            'Absences', 'Tenure_Years', 'Education'
        ]
        
        # Scale numeric features using the loaded scaler (if available)
        if scaler is not None:
            df[numeric_features] = scaler.transform(df[numeric_features])
        
        return df
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        print(f"Form data received: {form_data}")
        return None

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'employee_name': request.form['employee_name'],
                'employee_id': request.form['employee_id'],
                'gender': request.form['gender'],
                'age': request.form['age'],
                'marital_status': request.form['marital_status'],
                'education': request.form['education'],
                'position': request.form['position'],
                'years_of_service': request.form['years_of_service'],
                'annual_salary': request.form['annual_salary'],
                'recruitment_source': request.form['recruitment_source'],
                'engagement_score': request.form['engagement_score'],
                'satisfaction_score': request.form['satisfaction_score'],
                'special_projects': request.form['special_projects'],
                'training_hours': request.form['training_hours'],
                'absences': request.form['absences'],
                'days_late': request.form['days_late']
            }
            
            prediction_num = None
            
            # Try to use real model first, then fallback to dummy
            if model is not None and scaler is not None:
                preprocessed_data = preprocess_input(form_data)
                if preprocessed_data is not None:
                    prediction_num = model.predict(preprocessed_data)[0]
            
            # Fallback to dummy prediction
            if prediction_num is None:
                prediction_num = dummy_predict(form_data)
                print("Using dummy prediction")
            
            prediction_result = PERFORMANCE_LABELS.get(prediction_num, "Unknown Performance Level")
            prediction_message = f"Employee {form_data['employee_name'].strip()} is predicted as: {prediction_result}"
            
            # Save prediction with all form data to database
            save_prediction(form_data, prediction_result)
            
            # Render template with BOTH prediction and original form data
            return render_template('form.html', 
                                prediction_message=prediction_message,
                                form_data=form_data)
            
        except Exception as e:
            print(f"Error in prediction route: {e}")
            return render_template('form.html', 
                                prediction_message="Error occurred during prediction",
                                form_data=request.form if request.method == 'POST' else None)
    
    return render_template('form.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)

# API endpoint to get full prediction data for visualization
@app.route('/api/full_predictions')
def api_full_predictions():
    """API endpoint to get complete prediction data for visualization"""
    try:
        predictions = get_full_predictions_data()
        
        # Convert to list of dictionaries for JSON serialization
        prediction_list = []
        for pred in predictions:
            prediction_dict = {
                'employee_id': pred[0],
                'employee_name': pred[1],
                'gender': pred[2],
                'age': pred[3],
                'marital_status': pred[4],
                'education': pred[5],
                'position': pred[6],
                'years_of_service': pred[7],
                'annual_salary': pred[8],
                'recruitment_source': pred[9],
                'engagement_score': pred[10],
                'satisfaction_score': pred[11],
                'special_projects': pred[12],
                'training_hours': pred[13],
                'absences': pred[14],
                'days_late': pred[15],
                'predicted_performance': pred[16],
                'prediction_date': pred[17]
            }
            prediction_list.append(prediction_dict)
        
        return jsonify(prediction_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint to clear history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        clear_all_predictions()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Initialize database when app starts
if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    print("Starting Flask application...")
    app.run(debug=True)
    