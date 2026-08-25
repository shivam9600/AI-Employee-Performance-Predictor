import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ==========================================
# 1. Automatic Setup (Runs if files are missing)
# ==========================================
def initialize_system():
    """Creates the database and trains the model if they don't exist."""
    db_name = 'hr_analytics.db'
    
    # Check if we need to build the database and train the model
    if not os.path.exists(db_name) or not os.path.exists('performance_model.pkl'):
        with st.spinner("First-time setup: Building database and training AI model..."):
            # 1. Create DB and generate synthetic data
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    emp_id INTEGER PRIMARY KEY,
                    department TEXT,
                    years_experience INTEGER,
                    training_hours INTEGER,
                    projects_completed INTEGER,
                    salary REAL,
                    performance_score INTEGER
                )
            ''')
            
            # Generate 500 records
            np.random.seed(42)
            departments = ['Sales', 'Engineering', 'HR', 'Marketing']
            data = []
            for i in range(1, 501):
                dept = np.random.choice(departments)
                exp = np.random.randint(1, 15)
                training = np.random.randint(10, 100)
                projects = np.random.randint(2, 20)
                salary = 40000 + (exp * 3000) + np.random.randint(-5000, 5000)
                
                # Rule-based score generation for AI to learn
                score_base = (exp * 0.2) + (training * 0.05) + (projects * 0.5)
                if score_base > 15: perf = 3
                elif score_base > 8: perf = 2
                else: perf = 1
                data.append((i, dept, exp, training, projects, salary, perf))
                
            cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)', data)
            conn.commit()
            
            # 2. Train AI Model
            df = pd.read_sql_query("SELECT * FROM employees", conn)
            X = df.drop(['emp_id', 'performance_score'], axis=1)
            y = df['performance_score']
            
            le = LabelEncoder()
            X['department'] = le.fit_transform(X['department'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Save models
            joblib.dump(model, 'performance_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(le, 'label_encoder.pkl')
            conn.close()

# ==========================================
# 2. Page Configuration & Setup
# ==========================================
st.set_page_config(page_title="SkillCast | AI Performance", layout="wide")

# Run the initialization check
initialize_system()

st.title("🚀 SkillCast: AI Employee Performance Predictor")

# Load the trained ML models securely
model = joblib.load('performance_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# ==========================================
# 3. Database Helper Functions
# ==========================================
def get_all_employees():
    conn = sqlite3.connect('hr_analytics.db')
    df = pd.read_sql_query("SELECT * FROM employees ORDER BY emp_id DESC", conn)
    conn.close()
    return df

def save_new_employee(dept, exp, training, projects, salary, prediction):
    conn = sqlite3.connect('hr_analytics.db')
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(emp_id) FROM employees")
    max_id = cursor.fetchone()[0]
    new_id = (max_id + 1) if max_id else 1
    
    cursor.execute('''
        INSERT INTO employees (emp_id, department, years_experience, training_hours, projects_completed, salary, performance_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (new_id, dept, exp, training, projects, salary, int(prediction)))
    conn.commit()
    conn.close()
    return new_id

# ==========================================
# 4. User Interface
# ==========================================
tab1, tab2 = st.tabs(["🎯 Predict & Add Employee", "📊 View Database Records"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.markdown("### Enter New Employee Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            dept = st.selectbox("Department", ['Sales', 'Engineering', 'HR', 'Marketing'])
            exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
            salary = st.number_input("Yearly Salary ($)", min_value=30000, max_value=250000, value=65000, step=1000)
        with col2:
            training = st.number_input("Training Hours", min_value=0, max_value=300, value=40)
            projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=5)
        
        submitted = st.form_submit_button("Predict & Save to Database")

    if submitted:
        dept_encoded = le.transform([dept])[0]
        input_data = np.array([[dept_encoded, exp, training, projects, salary]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        new_id = save_new_employee(dept, exp, training, projects, salary, prediction)
        
        mapping = {
            1: '🔴 Low Performer', 
            2: '🟡 Average Performer', 
            3: '🟢 High Performer'
        }
        
        st.success(f"✅ Employee successfully added to Database with ID: {new_id}")
        st.markdown(f"### AI Prediction: {mapping[prediction]}")

# --- TAB 2: DATABASE DASHBOARD ---
with tab2:
    st.markdown("### Live Employee Database")
    df_employees = get_all_employees()
    
    score_mapping = {1: 'Low', 2: 'Average', 3: 'High'}
    df_employees['performance_score'] = df_employees['performance_score'].map(score_mapping)
    
    st.dataframe(df_employees, use_container_width=True)