# spam_classifier.py
import pickle
import string
import nltk
import google.generativeai as genai

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load model/vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Gemini setup (insert your actual API key)
genai.configure(api_key="AIzaSyBgFaA0Q14SUD5wznklfOYCXvvNIAUdsjI")
gemini = genai.GenerativeModel("gemini-pro")

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(word for word in text.split() if word not in stop_words)

def predict_spam_with_gemini(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]

    model_prediction = "Spam" if prob[1] > 0.5 else "Not Spam"
    confidence = max(prob)

    print(f"🔍 Model Prediction: {model_prediction} (confidence: {confidence:.2f})")

    # Trust model if very confident
    if confidence > 0.85:
        return model_prediction

    # Use Gemini for uncertain cases
    prompt = f"""
    You are a spam classification expert.
    Classify the following SMS message as either 'Spam' or 'Not Spam'.

    Message: "{message}"

    Respond with exactly one word: 'Spam' or 'Not Spam'.
    """
    try:
        response = gemini.generate_content(prompt)
        reply = response.text.strip().lower()
        print("🤖 Gemini Response:", reply)

        if "spam" in reply:
            return "Spam"
        elif "not spam" in reply:
            return "Not Spam"
        else:
            return model_prediction
    except Exception as e:
        print("Gemini Error:", e)
        return model_prediction
