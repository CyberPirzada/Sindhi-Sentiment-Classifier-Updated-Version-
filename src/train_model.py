import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/train.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(train["text"], train["label"])

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved to models/sentiment_model.pkl")
