import streamlit as st
import joblib
import re
import emoji

# ===============================
# LOAD MODEL
# ===============================

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===============================
# TEXT CLEANING
# ===============================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    return text


# ===============================
# EMOJI SENTIMENT
# ===============================

emoji_score = {
    "😀":1,"😃":1,"😄":1,"😁":1,
    "😊":1,"😍":2,"🥰":2,"😂":1,
    "🙂":1,
    "😐":0,"🤔":0,
    "😢":-1,"😭":-2,
    "😡":-2,"😠":-2
}

def extract_emojis(text):

    return ''.join(c for c in text if c in emoji.EMOJI_DATA)


def emoji_sentiment(emojis):

    score = 0

    for e in emojis:

        if e in emoji_score:

            score += emoji_score[e]

    if score > 0:
        return "positive"

    elif score < 0:
        return "negative"

    else:
        return "neutral"


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_sentiment(review):

    emojis = extract_emojis(review)

    # emoji-only input
    if emojis != "" and review.strip() == emojis:

        return emoji_sentiment(emojis)

    # text input
    cleaned = clean_text(review)

    X = vectorizer.transform([cleaned])

    sentiment = model.predict(X)[0]

    return sentiment


# ===============================
# STREAMLIT UI
# ===============================

st.title("🛒 Amazon Product Review Sentiment Classifier")

st.write("Enter a product review below to predict its sentiment (Negative / Neutral / Positive).")

review = st.text_area("✍ Enter Amazon Product Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:

        result = predict_sentiment(review)

        if result == "positive":
            st.success("Sentiment: Positive 😊")

        elif result == "negative":
            st.error("Sentiment: Negative 😡")

        else:
            st.info("Sentiment: Neutral 😐")