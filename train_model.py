# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, 'data', 'fake_job_postings.csv')
MODEL_PATH = os.path.join(ROOT, 'fake_job_model.pkl')
VECT_PATH = os.path.join(ROOT, 'tfidf_vectorizer.pkl')

# Load dataset or create small demo dataset if missing
if os.path.exists(DATA_PATH):
    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
else:
    print("No dataset found at data/fake_job_postings.csv — creating a small demo dataset.")
    df = pd.DataFrame({
        'description': [
            'Work from home, earn $5000 per week, no experience needed. Send bank details.',
            'Software Engineer required: 3+ years Python, remote possible. Apply with resume.',
            'Urgent hiring: pay per click, pay before training. Send payment now.',
            'Data analyst role, SQL and Excel required. Office location: Mumbai. Send CV.'
        ],
        'fraudulent': [1, 0, 1, 0]
    })

# Identify text and label columns
text_col = None
for c in ['job_description', 'description', 'full_description', 'job_post', 'text']:
    if c in df.columns:
        text_col = c; break
label_col = None
for c in ['fraudulent', 'fraud', 'is_fake', 'label', 'target']:
    if c in df.columns:
        label_col = c; break

if text_col is None or label_col is None:
    raise RuntimeError("Couldn't find text or label column. Expected columns like 'description' and 'fraudulent'.")

# Normalize labels to 0/1 (tries common encodings)
df = df[[text_col, label_col]].dropna().copy()
# If labels are strings like 'fake'/'real', map them
if df[label_col].dtype == object:
    df[label_col] = df[label_col].str.lower().map(lambda x: 1 if 'fake' in x or 'fraud' in x else 0)

df[label_col] = df[label_col].astype(int)

X_text = df[text_col].astype(str).values
y = df[label_col].values

# Print class distribution
(unique, counts) = np.unique(y, return_counts=True)
dist = dict(zip(unique, counts))
print("Label distribution (0=Real,1=Fake):", dist)

# Basic TF-IDF + LogisticRegression with balanced weights
vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_vect = vect.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluation
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

# Print some sample predictions from train and test to inspect behavior
def show_samples(texts, labels, name="SAMPLES", n=6):
    print(f"\n--- {name} (first {n}) ---")
    for i, (t, l) in enumerate(zip(texts[:n], labels[:n])):
        v = vect.transform([t])
        pred = clf.predict(v)[0]
        p = clf.predict_proba(v)[0] if hasattr(clf, "predict_proba") else None
        print(f"Text: {t[:120]!r}")
        print(f"  true: {l}  pred: {pred}  proba: {p}\n")

show_samples(X_text, y, "ALL DATA SAMPLES", n=6)

# Save model and vectorizer
joblib.dump(clf, MODEL_PATH)
joblib.dump(vect, VECT_PATH)
print("\nSaved model to:", MODEL_PATH)
print("Saved vectorizer to:", VECT_PATH)
