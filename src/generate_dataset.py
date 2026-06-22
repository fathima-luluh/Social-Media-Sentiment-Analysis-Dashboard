import pandas as pd
import random

positive = [
    "I love this product",
    "Amazing quality",
    "Excellent service",
    "Highly recommended",
    "Very satisfied",
    "Fast delivery",
    "Best app ever",
    "Fantastic experience"
]

negative = [
    "Worst experience",
    "Very disappointed",
    "Bad quality",
    "Terrible service",
    "Waste of money",
    "Never buying again",
    "Very frustrating",
    "Poor support"
]

neutral = [
    "It is okay",
    "Average experience",
    "Nothing special",
    "Works as expected",
    "Normal service",
    "Fine product",
    "Neutral opinion",
    "Acceptable quality"
]

rows = []

for _ in range(300):
    rows.append([random.choice(positive), "positive"])

for _ in range(300):
    rows.append([random.choice(negative), "negative"])

for _ in range(300):
    rows.append([random.choice(neutral), "neutral"])

df = pd.DataFrame(rows, columns=["text", "sentiment"])

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("data/raw/social_media_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())