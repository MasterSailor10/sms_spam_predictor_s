from flask import Flask, request, render_template, jsonify
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

app = Flask(__name__)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_sms = request.form.get('message')
        if not input_sms:
            return render_template('index.html', prediction='Please enter a message.')

        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            prediction = 'Spam'
        else:
            prediction = 'Not Spam'

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
