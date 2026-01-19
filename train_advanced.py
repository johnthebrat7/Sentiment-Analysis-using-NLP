"""
Advanced training with aggressive class balancing techniques.
This will fix the severe imbalance (92% positive, 8% negative).
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

print("\n" + "="*70)
print(" "*10 + "🔄 ADVANCED TRAINING WITH AGGRESSIVE BALANCING")
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
print(f"   ⚠️  SEVERE IMBALANCE: {pos_count/neg_count:.1f}:1 ratio\n")

# -------------------------------------------------
# OPTION 1: UNDERSAMPLE MAJORITY CLASS
# -------------------------------------------------
print("🔧 Applying undersampling to balance dataset...")

df_positive = data[data['feedback'] == 1]
df_negative = data[data['feedback'] == 0]

# Undersample positive class to 2x the negative class
# This keeps more data while balancing better
target_positive = min(len(df_negative) * 2, len(df_positive))

df_positive_sampled = resample(
    df_positive,
    n_samples=target_positive,
    random_state=42
)

# Combine
data_balanced = pd.concat([df_positive_sampled, df_negative])
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

new_pos = (data_balanced['feedback'] == 1).sum()
new_neg = (data_balanced['feedback'] == 0).sum()

print(f"✅ Balanced dataset:")
print(f"   Positive: {new_pos} ({new_pos/(new_pos+new_neg)*100:.1f}%)")
print(f"   Negative: {new_neg} ({new_neg/(new_pos+new_neg)*100:.1f}%)")
print(f"   New ratio: {new_pos/new_neg:.1f}:1\n")

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
print("🔄 Preprocessing balanced data...")
corpus = []
for idx, review in enumerate(data_balanced['verified_reviews']):
    corpus.append(preprocess_text(review))
    if (idx + 1) % 200 == 0:
        print(f"   Processed {idx + 1}/{len(data_balanced)}...")

print(f"✅ Preprocessing complete\n")

# -------------------------------------------------
# USE TFIDF INSTEAD OF COUNT VECTORIZER
# -------------------------------------------------
print("🔢 Using TF-IDF (better than CountVectorizer for imbalanced data)...")
tfidf = TfidfVectorizer(
    max_features=2500,
    ngram_range=(1, 2),  # Use bigrams for better context
    min_df=2,
    max_df=0.95
)
X = tfidf.fit_transform(corpus).toarray()
y = data_balanced['feedback'].values

print(f"✅ Feature matrix: {X.shape[0]} samples × {X.shape[1]} features\n")

# -------------------------------------------------
# TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# For models that need scaling
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"   Train split - Pos: {np.sum(y_train==1)}, Neg: {np.sum(y_train==0)}")
print(f"   Test split  - Pos: {np.sum(y_test==1)}, Neg: {np.sum(y_test==0)}\n")

# -------------------------------------------------
# SAVE COMPONENTS
# -------------------------------------------------
os.makedirs("Models", exist_ok=True)
pickle.dump(tfidf, open("Models/countVectorizer.pkl", "wb"))
pickle.dump(scaler, open("Models/scaler.pkl", "wb"))
print("💾 Saved TF-IDF vectorizer and scaler\n")

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
print("="*70)
print(" "*20 + "🤖 TRAINING MODELS")
print("="*70 + "\n")

results = {}

# Random Forest with stronger class weighting
print("[1/3] Random Forest (balanced + stronger parameters)...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced_subsample',  # Even stronger balancing
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_train = accuracy_score(y_train, rf.predict(X_train))
rf_test = accuracy_score(y_test, rf.predict(X_test))
rf_pred = rf.predict(X_test)

print(f"  Train: {rf_train*100:.2f}%, Test: {rf_test*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['Negative', 'Positive']))

results['Random Forest'] = {'model': rf, 'train': rf_train, 'test': rf_test}
pickle.dump(rf, open("Models/model_rf.pkl", "wb"))

# XGBoost with adjusted parameters
print("\n[2/3] XGBoost (with custom parameters)...")

# More aggressive scale_pos_weight
neg_train = np.sum(y_train == 0)
pos_train = np.sum(y_train == 1)
scale_weight = neg_train / pos_train

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scl, y_train, verbose=False)

xgb_train = accuracy_score(y_train, xgb.predict(X_train_scl))
xgb_test = accuracy_score(y_test, xgb.predict(X_test_scl))
xgb_pred = xgb.predict(X_test_scl)

print(f"  Train: {xgb_train*100:.2f}%, Test: {xgb_test*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=['Negative', 'Positive']))

results['XGBoost'] = {'model': xgb, 'train': xgb_train, 'test': xgb_test}
pickle.dump(xgb, open("Models/model_xgb.pkl", "wb"))

# Decision Tree
print("\n[3/3] Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train_scl, y_train)

dt_train = accuracy_score(y_train, dt.predict(X_train_scl))
dt_test = accuracy_score(y_test, dt.predict(X_test_scl))

print(f"  Train: {dt_train*100:.2f}%, Test: {dt_test*100:.2f}%")

results['Decision Tree'] = {'model': dt, 'train': dt_train, 'test': dt_test}
pickle.dump(dt, open("Models/model_dt.pkl", "wb"))

# -------------------------------------------------
# COMPARE AND SELECT BEST
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
# DETAILED TESTING
# -------------------------------------------------
print("="*70)
print(" "*20 + "🧪 COMPREHENSIVE TESTING")
print("="*70 + "\n")

test_cases = [
    ("This is absolutely terrible! Worst product ever! Complete waste of money!", "Negative"),
    ("I love this! Amazing quality and fast delivery! Highly recommend!", "Positive"),
    ("Horrible experience. Would not recommend to anyone. Very disappointed.", "Negative"),
    ("Great product! Highly recommend! Best purchase ever!", "Positive"),
    ("Awful. Broke after one day. Don't buy this junk.", "Negative"),
    ("Fantastic! Exceeded my expectations. Five stars!", "Positive"),
    ("Terrible quality. Not worth the money. Save yourself the trouble.", "Negative"),
    ("Perfect! Exactly what I needed. Works great!", "Positive"),
]

best_model = results[best_name]['model']
needs_scaling = best_name in ["XGBoost", "Decision Tree"]

correct = 0
neg_correct = 0
neg_total = 0
pos_correct = 0
pos_total = 0

for text, expected in test_cases:
    processed = preprocess_text(text)
    vec = tfidf.transform([processed]).toarray()
    
    if needs_scaling:
        vec = scaler.transform(vec)
    
    pred = best_model.predict(vec)[0]
    label = "Positive" if pred == 1 else "Negative"
    
    status = "✅" if label == expected else "❌"
    print(f"{status} Expected: {expected:<8} | Got: {label:<8} | '{text[:45]}...'")
    
    if label == expected:
        correct += 1
        if expected == "Negative":
            neg_correct += 1
        else:
            pos_correct += 1
    
    if expected == "Negative":
        neg_total += 1
    else:
        pos_total += 1

print(f"\n📊 Results:")
print(f"   Overall: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
print(f"   Negative: {neg_correct}/{neg_total} ({neg_correct/neg_total*100:.0f}%)")
print(f"   Positive: {pos_correct}/{pos_total} ({pos_correct/pos_total*100:.0f}%)")

print("\n" + "="*70)
print(" "*15 + "✅ ADVANCED TRAINING COMPLETE!")
print("="*70)
print("\n🚀 Now run: python app.py")
print("\n💡 Key improvements:")
print("   1. Undersampled majority class (2:1 ratio instead of 11:1)")
print("   2. Used TF-IDF with bigrams (better features)")
print("   3. Stronger class weighting in models")
print("   4. Optimized hyperparameters for imbalanced data")
print("="*70 + "\n")