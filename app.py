from flask import Flask, render_template, request, redirect, url_for, session
import joblib, sqlite3
import os
import pandas as pd
import csv
from datetime import datetime

app = Flask(__name__)

app.secret_key = "mysecretkey123"   # Required for session handling (keep secret)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fake_job_model.pkl')
VECT_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'predictions_log.csv')

# Load model & vectorizer 
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    print("Loaded model and vectorizer. Model classes:", getattr(model, "classes_", None))
except Exception as e:
    model = None
    vectorizer = None
    print("Warning: failed to load model/vectorizer. Run train_model.py. Error:", e)


@app.route('/')
def home():
    fake_count = 0
    real_count = 0

    conn = sqlite3.connect('job_predictions.db')
    cursor = conn.execute('SELECT prediction from predictions')
    for c in cursor:
        prediction = c[0]
        if prediction == 'Real Job':
            real_count += 1
        else:
            fake_count += 1
    
    '''if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        fake_count = (df['prediction'] == 'Fake Job').sum()
        real_count = (df['prediction'] == 'Real Job').sum()'''
        
    return render_template('index.html', fake_count=fake_count, real_count=real_count)

def append_log(job_description: str, prediction: str, confidence):
    header = ['timestamp', 'job_description', 'prediction', 'confidence']
    exists = os.path.exists(LOG_PATH)
    # Ensure directory exists 
    with open(LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        # write row 
        writer.writerow([timestamp, job_description, prediction, confidence])

@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form.get('job_description', '')
    if not job_desc or not job_desc.strip():
        return render_template('index.html', error="Please enter a job description.")

    if model is None or vectorizer is None:
        return render_template('index.html', error="Model files not found. Run the training script first.")

    # Transform input
    X_input = vectorizer.transform([job_desc])

    # Predict
    try:
        pred_raw = model.predict(X_input)[0]
    except Exception as e:
        print("Prediction error:", e)
        return render_template('index.html', error="Error during prediction. See server logs.")

    # Compute confidence mapped to predicted class if possible
    confidence = 'N/A'
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            classes = list(model.classes_)
            try:
                idx = classes.index(pred_raw)
            except ValueError:
                idx = int(probs.argmax())
                pred_raw = classes[idx]
            confidence = round(float(probs[idx]) * 100, 2)
        else:
            if hasattr(model, "decision_function"):
                score = model.decision_function(X_input)
                import numpy as _np
                conf_score = 1 / (1 + _np.exp(-float(score)))
                confidence = round(conf_score * 100, 2)
            else:
                confidence = 'N/A'
    except Exception as e:
        print("Confidence computation error:", e)
        confidence = 'N/A'

    # Normalize label to readable text
    try:
        pred_int = int(pred_raw)
        label = "Fake Job" if pred_int == 1 else "Real Job"
    except Exception:
        pred_str = str(pred_raw).lower()
        label = "Fake Job" if ('fake' in pred_str or 'fraud' in pred_str) else "Real Job"

    # Save to DB
    conn = sqlite3.connect('job_predictions.db')
    conn.execute('INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',
                 (job_desc, label, confidence))
    conn.commit()
    conn.close()

    # Append to CSV log
    '''try:
        append_log(job_desc, label, confidence)
    except Exception as e:
        print("Failed to append to log:", e)'''

    return render_template('result.html', label=label, confidence=confidence, description=job_desc)


@app.route('/history')
def history():
    # Fetching history from Database
    conn = sqlite3.connect('job_predictions.db')
    cursor = conn.execute('SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC')
    records = cursor.fetchall()
    conn.close()

    '''rows = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception as e:
            print("Failed to read log file:", e)
    # show latest first
    rows = list(reversed(rows))'''

    return render_template('history.html', records=records)

# ---------- ADMIN LOGIN PAGE ----------
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
 
        # Check credentials from SQLite
        conn = sqlite3.connect('job_predictions.db')
        cursor = conn.execute("SELECT * FROM admin WHERE username=? AND password=?", 
                             (username, password))
        admin = cursor.fetchone()
        conn.close()
 
        if admin:
            session['admin_logged_in'] = True
            return redirect('/admin_dashboard')
        else:
            return render_template('admin_login.html', error="Invalid username or password")
        
    return render_template('admin_login.html')

# ---------- ADMIN DASHBOARD (PROTECTED) ----------

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login') 
    # Fetch counts
    conn = sqlite3.connect('job_predictions.db')
    cursor = conn.cursor()

    fake_jobs = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    real_jobs = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    total = fake_jobs + real_jobs

    # Daily Count (group by date)
    daily_data = cursor.execute("""
        SELECT DATE(timestamp), COUNT(*)
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """).fetchall() 
    dates = [row[0] for row in daily_data]
    counts = [row[1] for row in daily_data]
    
    conn.close()
    
    return render_template('admin_dashboard.html', total=total, fake=fake_jobs, real=real_jobs, dates=dates, counts=counts)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/admin_login')

if __name__ == '__main__':
    app.run(debug=True)
