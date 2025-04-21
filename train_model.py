# train_model.py
import pandas as pd
import string
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load and preprocess
df = pd.read_csv("data/spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = LabelEncoder().fit_transform(df['label'])

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(word for word in text.split() if word not in stop_words)

df['message'] = df['message'].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved.")
