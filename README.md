SMS Spam Detector with Machine Learning + Gemini LLM
This project combines traditional machine learning with Google's Gemini large language model to
create a powerful SMS spam classifier. When the trained ML model lacks confidence, Gemini steps
in for accurate classification combining speed and intelligence for real-world usage.


Objectives
- Train a machine learning model to detect spam in SMS messages.
- Integrate Google's Gemini LLM to handle uncertain or edge cases.
- Build a simple web interface using Flask.
- Ensure high prediction accuracy on real-world text inputs.
- Deliver clean, modular, and scalable code.

Features
- ML Model: Trained using Naive Bayes on the Kaggle SMS Spam dataset.
- Gemini LLM: Handles general, modern, and tricky SMS patterns that ML may miss.
- Flask Web App: Input an SMS message and get a prediction instantly.
- Cleaned & lemmatized text preprocessing for higher accuracy.
- Fallback strategy for low-confidence predictions.


Project Structure
sms_spam_app/
 data/
 spam.csv
 model/
 spam_model.pkl
 vectorizer.pkl
 static/
 style.css
 templates/
 index.html
 app.py
 train_model.py
 spam_classifier.py
 requirements.txt


 
How to Run the Project
1. Clone the repository:
 git clone https://github.com/your-username/SpamDetection.git
 cd sms-spam-detector
2. Set up a virtual environment:(optional)
 python -m venv venv
 source venv/bin/activate # On Windows: venv\Scriptsctivate
3. Install dependencies:
 pip install -r requirements.txt
4. Download NLTK data:
 import nltk
 nltk.download('punkt')
 nltk.download('stopwords')
 nltk.download('wordnet')
5. Paste an api key from "https://aistudio.google.com/prompts/new_chat" > getapikey
6. Run the app:
 python app.py
Example
Input: "Congratulations! You've won a free cruise. Call now to claim."
Output: Spam
Notes
- Gemini LLM is only called when ML model is unsure (low confidence).
- Messages like "you won lottery" will now be correctly classified as Spam.
- The app is extensible for multilingual SMS or other LLMs.
Tech Stack
- Python, Scikit-learn, Flask, Google Gemini API, HTML + CSS, NLTK

