import joblib
from data_cleaning import clean_text

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

while True:
    text = input("\nEnter text (type exit to quit): ")

    if text.lower() == "exit":
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vector)[0]

    print("Predicted Sentiment:", prediction)