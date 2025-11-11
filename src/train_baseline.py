# src/train_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

df = pd.read_csv("data/sindhi_sentiment_dataset.csv", encoding="utf-8")
X = df["text"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=40000)),
    ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))

joblib.dump(pipe, "models/baseline_sentiment.pkl")
print("Saved baseline to models/baseline_sentiment.pkl")
