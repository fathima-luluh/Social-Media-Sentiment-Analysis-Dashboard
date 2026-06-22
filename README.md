\# 📊 Social Media Sentiment Analysis Dashboard



\## Overview



This project uses Natural Language Processing (NLP) and Machine Learning to classify social media comments into:



\- Positive 😀

\- Negative 😞

\- Neutral 😐



The results are visualized using an interactive Streamlit dashboard.



\---



\## Features



\- Text Cleaning

\- TF-IDF Feature Extraction

\- Logistic Regression Model

\- Single Text Prediction

\- CSV Upload Prediction

\- Pie Chart Visualization

\- Bar Chart Visualization

\- Download Predictions



\---



\## Tech Stack



\- Python

\- Pandas

\- NumPy

\- NLTK

\- Scikit-Learn

\- Streamlit

\- Plotly

\- Joblib



\---



\## Project Structure



```

Social-Media-Sentiment-Analysis-Dashboard

│

├── app

├── data

├── images

├── models

├── outputs

├── src

├── requirements.txt

├── README.md

└── .gitignore

```



\---



\## Run



Train Model:



```bash

python src/train\_model.py

```



Launch Dashboard:



```bash

streamlit run app/app.py

```



\---



\## Sample Predictions



| Text | Sentiment |

|--------|----------|

| I love this app | Positive |

| Worst service ever | Negative |

| Average experience | Neutral |



\---



\## Dashboard



Upload a CSV file or enter text manually to analyze sentiments.



\---



\## Author



Fathima Luluh

