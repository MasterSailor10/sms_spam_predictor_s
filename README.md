📩 SMS Spam Predictor

Detect Spam Messages with Machine Learning

🔗 Repository: MasterSailor10/sms_spam_predictor_s

🚀 Overview

The SMS Spam Predictor is a machine learning-based web application designed to detect spam messages efficiently. It uses Natural Language Processing (NLP) techniques to classify SMS messages as either Spam or Ham (Not Spam).

🛠️ Features

✔️ Machine Learning Model: Uses a pre-trained model stored in model.pkl for classification. ✔️ TF-IDF Vectorization: Converts text messages into numerical format using vectorizer.pkl. ✔️ Flask Web Application: A simple and interactive web interface to test SMS classification. ✔️ Fast & Efficient: Predicts in real-time with minimal latency.

📂 Project Structure

sms_spam_predictor_s/ │── spam_data/ # Dataset for training/testing
│── templates/ # HTML templates for the web app
│── app.py # Flask web application
│── model.pkl # Trained ML model
│── vectorizer.pkl # TF-IDF vectorizer
│── requirements.txt # Python dependencies
│── spam_data.zip # Compressed dataset
│── .gitignore # Ignored files
│── README.md # Project documentation

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/MasterSailor10/sms_spam_predictor_s.git cd sms_spam_predictor_s

2️⃣ Create & Activate Virtual Environment (Optional but Recommended)

python -m venv venv source venv/bin/activate # On macOS/Linux venv\Scripts\activate # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Application

python app.py Visit http://127.0.0.1:5000/ in your browser to access the web app.

🧪 Usage

1️⃣ Open the web application in your browser. 2️⃣ Enter an SMS message into the input field. 3️⃣ Click "Predict" to classify the message as Spam or Ham.

🎯 Model & Training

Dataset: The model is trained on an SMS spam dataset containing labeled messages. Preprocessing: Tokenization, stopword removal, and TF-IDF vectorization. Algorithm: Trained using a machine learning model (e.g., Naïve Bayes, SVM, or Logistic Regression).

🔗 Live Deployment

The application is deployed on Render for public access. Click here: https://sms-spam-predictor-ss.onrender.com

🤝 Contribution

Contributions are welcome! Feel free to fork this repository, create feature branches, and submit pull requests.

📜 License

This project is licensed under the MIT License – free to use and modify.

📬 Contact 👤 Siddhartha Singh 📧 10mastergaming10@gmail.com 🔗 https://www.linkedin.com/in/siddhartha1010/
