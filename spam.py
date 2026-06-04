import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# =============================================================================
# 1. TRAINING THE AI MODEL (Runs Automatically)
# =============================================================================

@st.cache_resource
def train_model():
    # Training Data
    data = {
        "label": [
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
            "ham", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"
        ],
        "message": [
            "Hey, are you coming to the party tonight?", "Congratulations! You won a free vacation. Reply WIN to claim.",
            "Can we meet tomorrow for lunch?", "You’ve been selected for a $500 gift card! Click the link now.",
            "Don’t forget our meeting at 3 PM.", "Win big prizes by joining our contest. Click here now!",
            "I’ll call you once I’m free.", "Your mobile number has won $10,000! Claim immediately.",
            "Happy Birthday! Hope you have an amazing day!", "Get cheap loans instantly. Apply online now.",
            "Let’s catch up this weekend.", "Dinner at 8?", "Exclusive offer! Get 70% off your next order.",
            "I’ll text you the address.", "You’ve won a new iPhone! Claim your reward today.",
            "How was your trip?", "You’ve been pre-approved for a credit card! Apply today.",
            "Movie night tonight?", "Please call me when you’re done.", "Congratulations! You’re our lucky winner!",
            "What time is the meeting?", "Urgent: Update your payment info now.", "See you tomorrow.",
            "Limited time offer: Buy one get one free.", "Are we still on for coffee?",
            "Your invoice is attached.", "Verify your account immediately to avoid closure.",
            "Lets go for a hike this weekend.", "Click here to download your gift.",
            "I sent you the files.", "YOU HAVE WON A CASH PRIZE!!!",
            "Call me.", "Act now! Free money waiting for you."
        ]
    }
    
    df = pd.DataFrame(data)
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X_vec, y)
    
    return model, vectorizer

model, vectorizer = train_model()

# =============================================================================
# 2. STREAMLIT WEBSITE INTERFACE
# =============================================================================

# Page Setup
st.set_page_config(page_title="Spam Detective", page_icon="🕵️‍♀️", layout="centered")

# Custom CSS
st.markdown("""
<style>
.stApp { background-color: #f8f9fc; }
h1 { color: #2c3e50; text-align: center; }
.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
div.stButton > button {
    width: 100%;
    background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    color: white;
    border: none;
    padding: 12px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 18px;
}
.spam-badge {
    background-color: #ff4b4b;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}
.ham-badge {
    background-color: #21c354;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Main Interface
st.title("🕵️‍♀️ Spam Detective")
st.markdown("<p class='subtitle'>AI-Powered SMS & Email Filter</p>", unsafe_allow_html=True)

st.markdown("### 📩 Enter Message Content")
message = st.text_area("Type your message here...", height=150, placeholder="e.g., 'Congratulations! You won a free vacation...'")

analyze_btn = st.button("🔍 Analyze Message")

# Prediction Logic
if analyze_btn:
    if message.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        text_vec = vectorizer.transform([message])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        
        st.markdown("---")
        st.subheader("🕵️‍♀️ Detection Result")
        
        confidence = probability[prediction] * 100
        
        if prediction == 1:
            st.markdown('<div class="spam-badge">⚠️ SPAM DETECTED</div>', unsafe_allow_html=True)
            st.error("This message contains malicious links or suspicious offers.")
        else:
            st.markdown('<div class="ham-badge">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)
            st.success("This message appears to be safe.")
        
        st.progress(int(confidence))
        st.caption(f"AI Confidence: {confidence:.2f}%")