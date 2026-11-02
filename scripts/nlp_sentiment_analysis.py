import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("1/3 Loading and cleaning customer reviews...")
reviews = pd.read_csv("data/raw/reviews.csv")

# Clean missing reviews and create binary sentiment target (1: Positive >= 4 stars, 0: Negative <= 3 stars)
reviews['review_text'] = reviews['review_text'].fillna("").astype(str).str.lower()
reviews['sentiment'] = (reviews['rating'] >= 4).astype(int)

# Save cleaned reviews
reviews.to_csv("data/processed/cleaned_reviews.csv", index=False)

print("2/3 Fitting TF-IDF Vectorizer and training Naive Bayes classifier...")
X = reviews['review_text']
y = reviews['sentiment']

vectorizer = TfidfVectorizer(stop_words='english', max_features=1500, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Quick evaluation
y_pred = model.predict(X_test)
print("\nClassification Summary:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

print("3/3 Exporting model artifacts to models/...")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(model, "models/sentiment_nb_model.pkl")

print("✓ Saved models/tfidf_vectorizer.pkl")
print("✓ Saved models/sentiment_nb_model.pkl")
