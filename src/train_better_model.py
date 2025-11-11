import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/sindhi_sentiment_dataset.csv")

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ML Pipeline (Better than basic model)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", RandomForestClassifier(n_estimators=200))
])

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

# Metrics
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to models/sentiment_model.pkl")
