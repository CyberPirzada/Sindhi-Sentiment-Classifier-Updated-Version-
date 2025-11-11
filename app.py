import streamlit as st
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------------------------------------
# Load Trained Model
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "models/transformer-xlm"
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Train the model first.")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

classifier = load_model()


# -----------------------------------------------------
# Detect Sindhi Language Function
# -----------------------------------------------------
def is_sindhi(text):
    # Sindhi unique characters (not present in Urdu/Pashto/Arabic)
    sindhi_specific_chars = "ٽٿڃڇڏڱڳٻڪ"
    return any(char in sindhi_specific_chars for char in text)

# -----------------------------------------------------
# ✅ UI Design
# -----------------------------------------------------

# Center logo using Streamlit layout
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")
with col2:
    st.image("images/logo.png", width=220)  # ✅ your renamed image
with col3:
    st.write("")

# Center Heading
st.markdown(
    "<h2 style='text-align:center; font-family:Arial;'>Sindhi Sentiment Classifier</h2>",
    unsafe_allow_html=True
)

st.write("Enter Sindhi text below ️")


# -----------------------------------------------------
# Text Input
# -----------------------------------------------------
user_text = st.text_area("✍ Enter Sindhi Text")


# -----------------------------------------------------
# Sentiment Prediction
# -----------------------------------------------------
if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("⚠ مهرباني ڪري ڪجهه لکو (Please enter text)")
    elif not is_sindhi(user_text):
        st.error("⚠ غلط ٻولي! صرف سنڌي لکو (Only Sindhi allowed)")
    else:
        result = classifier(user_text)[0]
        label = result["label"]
        score = result["score"]

        emoji_map = {
            "LABEL_0": "😡 Very Negative",
            "LABEL_1": "😔 Negative",
            "LABEL_2": "😐 Neutral",
            "LABEL_3": "🙂 Positive",
            "LABEL_4": "😍 Very Positive"
        }

        sentiment = emoji_map.get(label, label)

        st.success(f"Sentiment: {sentiment}")
        st.progress(score)
        st.caption(f"Confidence: {round(score * 100, 2)}%")
