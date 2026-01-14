# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import streamlit as st
# streamlit -> used to create interactive web applications

import numpy as np
# numpy -> used for numerical operations

from tensorflow.keras.models import load_model
# load_model -> used to load the trained RNN model

from tensorflow.keras.datasets import imdb
# imdb -> used to decode review text from word indices

from tensorflow.keras.preprocessing import sequence
# sequence -> used to pad input review


# =====================
# LOAD TRAINED MODEL
# =====================

model = load_model("simple_rnn_imdb.h5")
# Load the trained Simple RNN model


# =====================
# LOAD WORD INDEX
# =====================

word_index = imdb.get_word_index()
# Dictionary mapping words to integer indices

reverse_word_index = {
    value: key for key, value in word_index.items()
}
# Reverse mapping: index -> word


# =====================
# STREAMLIT APP TITLE
# =====================

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Predict whether a movie review is **Positive** or **Negative** :)")

st.markdown("---")


# =====================
# USER INPUT
# =====================

user_review = st.text_area(
    "Enter a movie review:",
    height=150
)
# Text area for user to input a movie review


# =====================
# PREPROCESS INPUT TEXT
# =====================

def preprocess_review(review):
    """
    Converts raw text review into padded sequence
    compatible with the trained RNN model
    """

    words = review.lower().split()
    # Convert text to lowercase and split into words

    encoded_review = [
        word_index.get(word, 2) for word in words
    ]
    # Convert words to indices
    # Unknown words are mapped to index 2

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=200
    )
    # Pad review to fixed length

    return padded_review


# =====================
# PREDICTION LOGIC
# =====================

if st.button("Analyze Sentiment >>"):

    if user_review.strip() == "":
        st.warning("Please enter a review first :|")

    else:
        processed_review = preprocess_review(user_review)
        # Preprocess user input

        prediction = model.predict(processed_review)
        # Predict sentiment probability

        probability = prediction[0][0]
        # Extract probability value

        st.write(f"Sentiment Probability: **{probability:.2f}**")

        if probability > 0.5:
            st.success("Positive Review :)")
        else:
            st.error("Negative Review :(")


# =====================
# APP FOOTER
# =====================

st.markdown("---")
st.caption(
    "Simple RNN | IMDB Dataset | TensorFlow / Keras"
)