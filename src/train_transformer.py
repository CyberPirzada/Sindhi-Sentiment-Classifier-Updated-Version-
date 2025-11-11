# src/train_transformer.py
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os

MODEL_NAME = "xlm-roberta-base"
OUT = "models/transformer-xlm"

# Load dataset
df = pd.read_csv("data/sindhi_sentiment_dataset.csv", encoding="utf-8")
df = df.sample(frac=1, random_state=42)  # shuffle

# Train / Val / Test split
train = df.sample(frac=0.85, random_state=42)
temp = df.drop(train.index)
val = temp.sample(frac=0.5, random_state=42)
test = temp.drop(val.index)

# Convert to HuggingFace datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val.reset_index(drop=True)),
    "test": Dataset.from_pandas(test.reset_index(drop=True))
})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(df["label"].unique())
)

# ✅ FIX compute_metrics to return floats
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=OUT,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none"  # disables wandb etc logs
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train & Save
trainer.train()
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)

print("✅ Training complete. Model saved in:", OUT)
