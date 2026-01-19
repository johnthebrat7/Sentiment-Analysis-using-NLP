"""
Quick retraining script with class balancing.
This will fix the bias issue by giving equal weight to positive and negative examples.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "="*70)
print(" "*15 + "🔄 RETRAINING WITH CLASS BALANCING")
print("="*70 + "\n")

# -------------------------------------------------
# SETUP
# -------------------------------------------------
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return ' '.join(text)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
print("📂 Loading dataset...")
data_path = r"D:\BackEndProjects\SentimentAnalysisSelf\data\amazon_alexa.tsv"

data = pd.read_csv(data_path, delimiter="\t", quoting=3)
data.dropna(inplace=True)

pos_count = (data['feedback'] == 1).sum()
neg_count = (data['feedback'] == 0).sum()
total = len(data)

print(f"✅ Loaded {total} reviews")
print(f"   Positive: {pos_count} ({pos_count/total*100:.1f}%)")
print(f"   Negative: {neg_count} ({neg_count/total*100:.1f}%)")

if pos_count / total > 0.85:
    print(f"   ⚠️  Dataset is HIGHLY imbalanced!")
    print(f"   → Using class_weight='balanced' to fix this\n")
else:
    print(f"   ✅ Dataset balance is acceptable\n")

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
print("🔄 Preprocessing...")
corpus = []
for idx, review in enumerate(data['verified_reviews']):
    corpus.append(preprocess_text(review))
    if (idx + 1) % 500 == 0:
        print(f"   Processed {idx + 1}/{len(data)}...")

print(f"✅ Preprocessing complete\n")

# -------------------------------------------------
# VECTORIZE & SCALE
# -------------------------------------------------
print("🔢 Extracting features...")
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

# -------------------------------------------------
# SAVE VECTORIZER & SCALER
# -------------------------------------------------
os.makedirs("Models", exist_ok=True)
pickle.dump(cv, open("Models/countVectorizer.pkl", "wb"))
pickle.dump(scaler, open("Models/scaler.pkl", "wb"))
print("💾 Saved vectorizer and scaler\n")

# -------------------------------------------------
# TRAIN MODELS WITH CLASS BALANCING
# -------------------------------------------------
print("="*70)
print(" "*20 + "🤖 TRAINING MODELS")
print("="*70 + "\n")

results = {}

# Random Forest (with class balancing)
print("[1/3] Random Forest (with class_weight='balanced')...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    class_weight='balanced',  # KEY FIX!
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)  # RF doesn't need scaling

rf_train = accuracy_score(y_train, rf.predict(X_train))
rf_test = accuracy_score(y_test, rf.predict(X_test))

print(f"  Train: {rf_train*100:.2f}%, Test: {rf_test*100:.2f}%")
results['Random Forest'] = {'model': rf, 'train': rf_train, 'test': rf_test}
pickle.dump(rf, open("Models/model_rf.pkl", "wb"))

# XGBoost (with scale_pos_weight)
print("\n[2/3] XGBoost (with scale_pos_weight)...")
scale_weight = neg_count / pos_count
print(f"  Using scale_pos_weight: {scale_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_weight,  # KEY FIX!
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scl, y_train, verbose=False)

xgb_train = accuracy_score(y_train, xgb.predict(X_train_scl))
xgb_test = accuracy_score(y_test, xgb.predict(X_test_scl))

print(f"  Train: {xgb_train*100:.2f}%, Test: {xgb_test*100:.2f}%")
results['XGBoost'] = {'model': xgb, 'train': xgb_train, 'test': xgb_test}
pickle.dump(xgb, open("Models/model_xgb.pkl", "wb"))

# Decision Tree (with class balancing)
print("\n[3/3] Decision Tree (with class_weight='balanced')...")
dt = DecisionTreeClassifier(
    max_depth=10,
    class_weight='balanced',  # KEY FIX!
    random_state=42
)
dt.fit(X_train_scl, y_train)

dt_train = accuracy_score(y_train, dt.predict(X_train_scl))
dt_test = accuracy_score(y_test, dt.predict(X_test_scl))

print(f"  Train: {dt_train*100:.2f}%, Test: {dt_test*100:.2f}%")
results['Decision Tree'] = {'model': dt, 'train': dt_train, 'test': dt_test}
pickle.dump(dt, open("Models/model_dt.pkl", "wb"))

# -------------------------------------------------
# SELECT BEST MODEL
# -------------------------------------------------
print("\n" + "="*70)
print(" "*20 + "📊 MODEL COMPARISON")
print("="*70 + "\n")

print(f"{'Model':<20} {'Train':>12} {'Test':>12} {'Overfit':>12}")
print("-" * 70)

best_name = None
best_test = 0

for name, metrics in results.items():
    train = metrics['train']
    test = metrics['test']
    overfit = train - test
    print(f"{name:<20} {train*100:>11.2f}% {test*100:>11.2f}% {overfit*100:>11.2f}%")
    
    if test > best_test:
        best_test = test
        best_name = name

print(f"\n🏆 BEST: {best_name} ({best_test*100:.2f}%)")

# Save best model
pickle.dump(results[best_name]['model'], open("sentiment_model.pkl", "wb"))
print(f"💾 Saved as sentiment_model.pkl\n")

# -------------------------------------------------
# TEST WITH SAMPLE TEXTS
# -------------------------------------------------
print("="*70)
print(" "*20 + "🧪 TESTING PREDICTIONS")
print("="*70 + "\n")

test_texts = [
    ("This is terrible! Worst product ever!", "Negative"),
    ("I love this! Amazing quality!", "Positive"),
    ("Horrible experience. Would not recommend.", "Negative"),
    ("Great product! Highly recommend!", "Positive"),
]

best_model = results[best_name]['model']
needs_scaling = best_name in ["XGBoost", "Decision Tree"]

correct = 0
for text, expected in test_texts:
    processed = preprocess_text(text)
    vec = cv.transform([processed]).toarray()
    
    if needs_scaling:
        vec = scaler.transform(vec)
    
    pred = best_model.predict(vec)[0]
    label = "Positive" if pred == 1 else "Negative"
    
    status = "✅" if label == expected else "❌"
    print(f"{status} '{text[:40]}...'")
    print(f"   Expected: {expected}, Got: {label}\n")
    
    if label == expected:
        correct += 1

print(f"Accuracy on test cases: {correct}/{len(test_texts)} ({correct/len(test_texts)*100:.0f}%)")

print("\n" + "="*70)
print(" "*20 + "✅ RETRAINING COMPLETE!")
print("="*70)
print("\n🚀 Now run: python app.py")
print("="*70 + "\n")