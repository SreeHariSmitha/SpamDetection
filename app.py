from flask import Flask, render_template, request
from spam_classifier import predict_spam_with_gemini

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        sms = request.form.get('sms')
        if sms:
            result = predict_spam_with_gemini(sms)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
