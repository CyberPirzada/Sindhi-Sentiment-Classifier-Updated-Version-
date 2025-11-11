import pandas as pd
import re


def auto_label(text):
    positive_words = ["سٺو", "بهتر", "بهترين", "لاجواب", "زبردست", "پسند", "خوش", "پيار", "عظيم", "شاندار", "مزو"]
    negative_words = ["خراب", "بد", "بدتر", "بيڪار", "ناڪام", "مايوس", "نفرت", "غلط", "سست", "فضول", "ڏک", "تڪليف"]

    text = text.lower()

    pos_count = sum(word in text for word in positive_words)
    neg_count = sum(word in text for word in negative_words)

    if pos_count > neg_count:
        return 2  # Positive
    elif neg_count > pos_count:
        return 0  # Negative
    else:
        return 1  # Neutral


def clean_text(txt):
    txt = re.sub(r"http\S+","",txt)
    txt = re.sub(r"[^\u0600-\u06FF ]","",txt)
    txt = re.sub(r"\s+"," ",txt).strip()
    return txt

df = pd.read_csv("data/sindhi_sentiment_dataset.csv")
df["text"] = df["text"].apply(clean_text)
df["label"] = df["text"].apply(auto_label)
df.to_csv("data/processed/cleaned.csv", index=False, encoding="utf-8-sig")
print("✅ Cleaned data saved!")
