"""
Diagnostic script to identify why all predictions are Positive.
This will check preprocessing, vectorization, and model predictions step-by-step.
"""

import os
import re
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

print("\n" + "="*70)
print(" "*20 + "🔍 DIAGNOSTIC MODE")
print("="*70 + "\n")

# -------------------------------------------------
# SETUP
# -------------------------------------------------
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text exactly as in training."""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return ' '.join(text)

# -------------------------------------------------
# LOAD COMPONENTS
# -------------------------------------------------
print("📦 Loading components...\n")

try:
    vectorizer = pickle.load(open('models/countVectorizer.pkl', 'rb'))
    print("✅ Vectorizer loaded")
except Exception as e:
    print(f"❌ Vectorizer failed: {e}")
    exit(1)

try:
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print("✅ Scaler loaded")
except Exception as e:
    print(f"❌ Scaler failed: {e}")
    scaler = None

try:
    model_rf = pickle.load(open('models/model_rf.pkl', 'rb'))
    print("✅ Random Forest loaded")
    model = model_rf
    model_name = "Random Forest"
    needs_scaling = False
except:
    model_rf = None

try:
    model_xgb = pickle.load(open('models/model_xgb.pkl', 'rb'))
    print("✅ XGBoost loaded")
    if model_rf is None:
        model = model_xgb
        model_name = "XGBoost"
        needs_scaling = True
except:
    model_xgb = None

try:
    model_dt = pickle.load(open('models/model_dt.pkl', 'rb'))
    print("✅ Decision Tree loaded")
    if model_rf is None and model_xgb is None:
        model = model_dt
        model_name = "Decision Tree"
        needs_scaling = True
except:
    model_dt = None

print(f"\n🤖 Using: {model_name}")
print(f"📊 Scaling needed: {needs_scaling}\n")

# -------------------------------------------------
# TEST CASES
# -------------------------------------------------
test_cases = [
    ("This is absolutely terrible! Worst product ever!", "Negative"),
    ("I love this! Amazing quality and fast delivery!", "Positive"),
    ("Horrible experience. Would not recommend.", "Negative"),
    ("Great product! Highly recommend!", "Positive"),
    ("Waste of money. Broke after one day.", "Negative"),
]

print("="*70)
print(" "*20 + "🧪 RUNNING DIAGNOSTICS")
print("="*70 + "\n")

for i, (text, expected) in enumerate(test_cases, 1):
    print(f"Test {i}: {expected.upper()} sentiment")
    print(f"Input: \"{text}\"")
    print("-" * 70)
    
    # Step 1: Preprocess
    processed = preprocess_text(text)
    print(f"1. Preprocessed: \"{processed}\"")
    
    # Step 2: Vectorize
    vectorized = vectorizer.transform([processed]).toarray()
    print(f"2. Vectorized shape: {vectorized.shape}")
    print(f"   Non-zero features: {np.count_nonzero(vectorized)}")
    print(f"   Feature sum: {np.sum(vectorized):.2f}")
    
    # Step 3: Scale (if needed)
    if needs_scaling and scaler is not None:
        scaled = scaler.transform(vectorized)
        print(f"3. Scaled shape: {scaled.shape}")
        print(f"   Min value: {np.min(scaled):.4f}")
        print(f"   Max value: {np.max(scaled):.4f}")
        X_final = scaled
    else:
        print("3. Scaling: SKIPPED (Random Forest doesn't need it)")
        X_final = vectorized
    
    # Step 4: Predict
    prediction = model.predict(X_final)[0]
    pred_label = "Positive" if prediction == 1 else "Negative"
    
    print(f"4. Raw prediction: {prediction}")
    print(f"   Prediction label: {pred_label}")
    
    # Step 5: Probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_final)[0]
        print(f"5. Probabilities: Negative={proba[0]:.4f}, Positive={proba[1]:.4f}")
        confidence = max(proba) * 100
        print(f"   Confidence: {confidence:.2f}%")
    
    # Result
    if pred_label == expected:
        print(f"\n✅ CORRECT: Expected {expected}, Got {pred_label}")
    else:
        print(f"\n❌ WRONG: Expected {expected}, Got {pred_label}")
    
    print("\n" + "="*70 + "\n")

# -------------------------------------------------
# CHECK MODEL CLASS DISTRIBUTION
# -------------------------------------------------
print("="*70)
print(" "*20 + "📊 MODEL ANALYSIS")
print("="*70 + "\n")

# Check if model always predicts the same class
print("Testing if model is biased towards one class...\n")

# Create some diverse test vectors
random_tests = []
for _ in range(20):
    random_vec = np.random.rand(1, vectorizer.max_features if hasattr(vectorizer, 'max_features') else 2500)
    if needs_scaling and scaler is not None:
        random_vec = scaler.transform(random_vec)
    pred = model.predict(random_vec)[0]
    random_tests.append(pred)

positive_count = sum(random_tests)
negative_count = len(random_tests) - positive_count

print(f"Random predictions (20 tests):")
print(f"  Positive: {positive_count} ({positive_count/20*100:.1f}%)")
print(f"  Negative: {negative_count} ({negative_count/20*100:.1f}%)")

if positive_count == 20:
    print("\n⚠️  WARNING: Model ALWAYS predicts Positive!")
    print("   Possible causes:")
    print("   1. Model wasn't trained properly")
    print("   2. Training data was imbalanced")
    print("   3. Preprocessing mismatch between training and prediction")
elif negative_count == 20:
    print("\n⚠️  WARNING: Model ALWAYS predicts Negative!")
else:
    print("\n✅ Model can predict both classes")

# -------------------------------------------------
# CHECK VECTORIZER VOCABULARY
# -------------------------------------------------
print("\n" + "="*70)
print(" "*20 + "📚 VECTORIZER VOCABULARY CHECK")
print("="*70 + "\n")

if hasattr(vectorizer, 'vocabulary_'):
    vocab_size = len(vectorizer.vocabulary_)
    print(f"Vocabulary size: {vocab_size}")
    
    # Check if common sentiment words are in vocabulary
    sentiment_words = {
        'Positive': ['love', 'great', 'excellent', 'amazing', 'good', 'best'],
        'Negative': ['hate', 'terrible', 'awful', 'worst', 'bad', 'horrible']
    }
    
    for sentiment, words in sentiment_words.items():
        print(f"\n{sentiment} words in vocabulary:")
        for word in words:
            # Stem the word
            stemmed = stemmer.stem(word)
            if stemmed in vectorizer.vocabulary_:
                print(f"  ✅ '{word}' (stemmed: '{stemmed}')")
            else:
                print(f"  ❌ '{word}' (stemmed: '{stemmed}') - NOT FOUND")

# -------------------------------------------------
# RECOMMENDATION
# -------------------------------------------------
print("\n" + "="*70)
print(" "*25 + "💡 RECOMMENDATIONS")
print("="*70 + "\n")

if positive_count == 20 or negative_count == 20:
    print("🔴 CRITICAL ISSUE DETECTED:")
    print("   Your model is biased and only predicts one class.\n")
    print("Solutions:")
    print("  1. Re-run train_model.py to retrain the model")
    print("  2. Check if your dataset has balanced classes (50/50 split)")
    print("  3. Verify preprocessing is identical in training and prediction")
    print("\nRun this command:")
    print("  python train_model.py")
else:
    print("Model seems capable of predicting both classes.")
    print("If you're still getting wrong predictions, the issue might be:")
    print("  1. Preprocessing differences")
    print("  2. Model needs retraining with better parameters")
    print("  3. Feature extraction settings")

print("\n" + "="*70 + "\n")