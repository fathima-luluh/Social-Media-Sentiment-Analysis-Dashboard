import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_cleaning import clean_text

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
data = pd.read_csv("data/raw/social_media_data.csv")

print("\nDataset Shape:")
print(data.shape)

print("\nClass Distribution:")
print(data["sentiment"].value_counts())

print("\nDataset Loaded:")
print(data.head())

# -----------------------------------
# 2. Clean Text
# -----------------------------------
data["cleaned_text"] = data["text"].apply(clean_text)

# -----------------------------------
# 3. Features and Labels
# -----------------------------------
X = data["cleaned_text"]
y = data["sentiment"]

# -----------------------------------
# 4. TF-IDF Vectorization
# -----------------------------------
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)

# -----------------------------------
# 5. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 6. Train Model
# -----------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------------
# 7. Predictions
# -----------------------------------
y_pred = model.predict(X_test)

print("\nActual Labels:")
print(list(y_test))

print("\nPredicted Labels:")
print(list(y_pred))

# -----------------------------------
# 8. Evaluation
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("------------------")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# 9. Save Model & Vectorizer
# -----------------------------------
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")