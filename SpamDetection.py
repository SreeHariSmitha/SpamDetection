import pandas as pd
import string
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# 1. Load the dataset from Kaggle CSV
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns

# 2. Encode labels: 'ham' → 0, 'spam' → 1
df['label'] = LabelEncoder().fit_transform(df['label'])

# 3. Clean text messages
def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([c for c in text if c not in string.punctuation])  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

df['message'] = df['message'].apply(clean_text)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 5. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Function to classify a new message
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vec)[0]
    return "Spam" if pred == 1 else "Not Spam"

# 8. Test with your own input
sample = input("Enter the text :")
print(f"Prediction: {predict_message(sample)}")

