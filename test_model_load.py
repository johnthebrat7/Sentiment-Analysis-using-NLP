import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

# ---------------------------
# NLTK SETUP
# ---------------------------
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv(
    r"D:\ComputerScience\SentimentAnalysisProject\amazon_alexa.tsv",
    delimiter="\t",
    quoting=3
)

data.dropna(inplace=True)

# ---------------------------
# TEXT PREPROCESSING
# ---------------------------
corpus = []
for review in data['verified_reviews']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    corpus.append(' '.join(review))

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------------
# SCALING (Only for XGBoost / DT)
# ---------------------------
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# ---------------------------
# CREATE MODEL DIRECTORY
# ---------------------------
os.makedirs("Models", exist_ok=True)

pickle.dump(cv, open("Models/countVectorizer.pkl", "wb"))
pickle.dump(scaler, open("Models/scaler.pkl", "wb"))

# ---------------------------
# RANDOM FOREST + GRID SEARCH
# ---------------------------
rf = RandomForestClassifier(random_state=42)

params = {
    "n_estimators": [200, 300],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    rf,
    params,
    cv=cv_strategy,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("Best RF Accuracy:", accuracy_score(y_test, best_rf.predict(X_test)))

pickle.dump(best_rf, open("Models/model_rf.pkl", "wb"))

# ---------------------------
# XGBOOST
# ---------------------------
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

xgb.fit(X_train_scl, y_train)

print("XGB Accuracy:", accuracy_score(y_test, xgb.predict(X_test_scl)))

pickle.dump(xgb, open("Models/model_xgb.pkl", "wb"))

# ---------------------------
# DECISION TREE
# ---------------------------
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train_scl, y_train)

print("DT Accuracy:", accuracy_score(y_test, dt.predict(X_test_scl)))

pickle.dump(dt, open("Models/model_dt.pkl", "wb"))
