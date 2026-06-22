import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Social Media Sentiment Analysis Dashboard",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Social Media Sentiment Analysis Dashboard")
st.write("Analyze comments and classify them into Positive, Negative, or Neutral sentiments.")

option = st.radio(
    "Choose Input Type",
    ["Single Text", "Upload CSV"]
)

# ---------------------------
# Single Text Prediction
# ---------------------------
if option == "Single Text":

    text = st.text_area("Enter text")

    if st.button("Predict Sentiment"):

        if text.strip() != "":
            vector = vectorizer.transform([text])
            prediction = model.predict(vector)[0]

            st.success(f"Predicted Sentiment: {prediction}")

# ---------------------------
# CSV Upload
# ---------------------------
else:

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")

        else:

            vectors = vectorizer.transform(df["text"])
            df["Predicted_Sentiment"] = model.predict(vectors)

            st.subheader("Prediction Results")
            st.dataframe(df)

            sentiment_counts = (
                df["Predicted_Sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_counts.columns = ["Sentiment", "Count"]

            # Pie Chart
            fig1 = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig1)

            # Bar Chart
            fig2 = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Count",
                title="Sentiment Count"
            )

            st.plotly_chart(fig2)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )