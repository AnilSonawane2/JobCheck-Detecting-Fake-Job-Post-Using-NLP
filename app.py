from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fake_job_model.pkl')
VECT_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'predictions_log.csv')

# Load model & vectorizer (fail gracefully)
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
    return render_template('index.html')

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

    # Append to CSV log
    try:
        append_log(job_desc, label, confidence)
    except Exception as e:
        print("Failed to append to log:", e)

    return render_template('result.html', label=label, confidence=confidence, description=job_desc)


@app.route('/history')
def history():
    rows = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception as e:
            print("Failed to read log file:", e)
    # show latest first
    rows = list(reversed(rows))
    return render_template('history.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
