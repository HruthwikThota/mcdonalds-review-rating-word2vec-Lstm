
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import json
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="McDonald's Review Rating Prediction",
    page_icon="🍔",
    layout="centered"
)

# Project Directory
BASE_DIR = Path(__file__).parent

# Load Trained Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        BASE_DIR / "final_word2vec_lstm_model.keras"
    )
    return model

# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    with open(BASE_DIR / "tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Load Model Configuration
@st.cache_data
def load_config():
    with open(BASE_DIR / "best_lstm_hyperparameters.json", "r") as file:
        config = json.load(file)
    return config

model = load_model()
tokenizer = load_tokenizer()
config = load_config()

maxlen = config["maxlen"]

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-z'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction Function
def predict_rating(review):

    cleaned_review = clean_text(review)

    if not cleaned_review:
        return None, None

    # Convert Text into Sequence
    sequence = tokenizer.texts_to_sequences([cleaned_review])

    # Apply Padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=maxlen,
        padding='post',
        truncating='post'
    )

    # Predict Rating
    prediction = model.predict(padded_sequence, verbose=0)

    predicted_class = np.argmax(prediction, axis=1)[0]

    predicted_rating = int(predicted_class) + 1

    confidence = float(np.max(prediction)) * 100

    return predicted_rating, confidence


# Application Interface
st.title("🍔 McDonald's Review Rating Prediction")

st.write(
    "Enter a customer review below. "
    "The AI model predicts the customer's rating "
    "using Word2Vec embeddings and LSTM."
)

st.divider()

# User Input
review = st.text_area(
    "Enter Customer Review",
    placeholder="Example: The food was delicious and the service was excellent...",
    height=150
)

# Prediction Button
if st.button("Predict Rating", type="primary", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a customer review.")

    else:
        with st.spinner("Analyzing customer review..."):

            rating, confidence = predict_rating(review)

        if rating is None:
            st.warning("Please enter a valid customer review.")

        else:
            st.success("Prediction Completed!")

            st.subheader("Predicted Customer Rating")

            st.markdown(f"# {'⭐' * rating}")

            st.metric(
                label="Predicted Rating",
                value=f"{rating} / 5 Stars"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            if rating <= 2:
                st.error("Negative Customer Experience")

            elif rating == 3:
                st.info("Neutral Customer Experience")

            else:
                st.success("Positive Customer Experience")

st.divider()

st.caption(
    "Model: Word2Vec + LSTM | "
    "Dataset: McDonald's Customer Reviews | "
    "Hyperparameter Tuning: Optuna"
)
