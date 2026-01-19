from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pickle
import re
import numpy as np

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = "supersecretkey"

# -------------------------------------------------
# PATH CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "countVectorizer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "model_xgb.pkl")
DT_PATH = os.path.join(MODEL_DIR, "model_dt.pkl")

# -------------------------------------------------
# PREPROCESSING SETUP
# -------------------------------------------------
stemmer = PorterStemmer()
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text exactly as done during training.
    Steps: Remove special chars → Lowercase → Stem → Remove stopwords
    """
    # Remove special characters (keep only letters)
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Lowercase and split
    text = text.lower().split()
    # Stem and remove stopwords
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    # Join back to string
    return ' '.join(text)

# -------------------------------------------------
# LOAD ARTIFACTS SAFELY
# -------------------------------------------------
def safe_load(path, name):
    """Load pickle file safely with error handling."""
    if not os.path.exists(path):
        app.logger.warning(f"{name} not found at {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        app.logger.error(f"Failed loading {name}: {e}")
        return None


vectorizer = safe_load(VECTORIZER_PATH, "Vectorizer")
scaler = safe_load(SCALER_PATH, "Scaler")
model_rf = safe_load(RF_PATH, "RandomForest")
model_xgb = safe_load(XGB_PATH, "XGBoost")
model_dt = safe_load(DT_PATH, "DecisionTree")

# -------------------------------------------------
# SELECT BEST MODEL
# -------------------------------------------------
MODEL_TYPE = None
model = None

# Priority: RF > XGB > DT (based on typical performance)
if model_rf is not None:
    model = model_rf
    MODEL_TYPE = "Random Forest"
elif model_xgb is not None:
    model = model_xgb
    MODEL_TYPE = "XGBoost"
elif model_dt is not None:
    model = model_dt
    MODEL_TYPE = "Decision Tree"

# -------------------------------------------------
# VALIDATE REQUIRED COMPONENTS
# -------------------------------------------------
if model is None or vectorizer is None:
    print("\n" + "="*70)
    print(" "*20 + "⚠️  CRITICAL ERROR")
    print("="*70)
    print("\n❌ Required ML components are missing!")
    print("\nMissing components:")
    if model is None:
        print("  ❌ Model files (model_rf.pkl, model_xgb.pkl, or model_dt.pkl)")
    if vectorizer is None:
        print("  ❌ Vectorizer (countVectorizer.pkl)")
    print("\n💡 Solution: Run 'python train_model.py' to create these files.")
    print("="*70 + "\n")
    raise RuntimeError("❌ Critical ML artifacts missing. Flask cannot start.")

if scaler is None:
    app.logger.warning("⚠️  Scaler not found - will skip scaling step")

app.logger.info(f"✅ Using model: {MODEL_TYPE}")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def get_label(pred):
    """Convert prediction (0/1) to readable label."""
    return "Positive" if int(pred) == 1 else "Negative"

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    """Home page with input form."""
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Process text and return sentiment prediction."""
    text = request.form.get("text", "").strip()

    if not text or len(text) < 3:
        flash("Please enter valid text (minimum 3 characters)", "warning")
        return redirect(url_for("home"))

    try:
        # Step 1: Preprocess text
        processed = preprocess_text(text)
        app.logger.info(f"Preprocessed: {processed[:50]}...")

        # Step 2: Vectorize
        X_vec = vectorizer.transform([processed]).toarray()

        # Step 3: Scale (ONLY for XGBoost / Decision Tree)
        if MODEL_TYPE in ["XGBoost", "Decision Tree"] and scaler is not None:
            X_vec = scaler.transform(X_vec)
            app.logger.info("Applied scaling")

        # Step 4: Predict
        prediction = model.predict(X_vec)[0]
        label = get_label(prediction)

        # Step 5: Get confidence
        confidence = 85.0
        if hasattr(model, "predict_proba"):
            confidence = round(np.max(model.predict_proba(X_vec)[0]) * 100, 2)

        app.logger.info(f"Prediction: {label} ({confidence}%)")

        return render_template(
            "result.html",
            text=text,
            prediction=label,
            confidence=confidence,
            model_used=MODEL_TYPE
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        flash(f"Error during prediction: {str(e)}", "error")
        return redirect(url_for("home"))

@app.route("/about")
def about():
    """About page with project information."""
    return render_template("about.html")

# -------------------------------------------------
# ERROR HANDLERS
# -------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Internal server error"), 500

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "🎭 SENTIMENT ANALYSIS APP")
    print("="*70 + "\n")
    
    # Print component status
    print("📦 Component Status:")
    print(f"  {'✅' if vectorizer else '❌'} Vectorizer: {VECTORIZER_PATH}")
    print(f"  {'✅' if scaler else '⚠️ '} Scaler:     {SCALER_PATH}")
    print(f"  {'✅' if model_rf else '❌'} Random Forest: {RF_PATH}")
    print(f"  {'✅' if model_xgb else '❌'} XGBoost:       {XGB_PATH}")
    print(f"  {'✅' if model_dt else '❌'} Decision Tree: {DT_PATH}")
    
    print(f"\n🤖 Active Model: {MODEL_TYPE}")
    print("\n🚀 Starting Flask server...")
    print("   Visit: http://127.0.0.1:5000")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True)