import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure NLTK data packages are downloaded
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']:
    nltk.download(resource, quiet=True)

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("1/5 Loading raw review dataset...")
reviews = pd.read_csv("data/raw/reviews.csv")
print(f"Total reviews loaded: {len(reviews)}")

# Binary Sentiment Ground Truth:
# Rating 4 or 5 -> Positive (1)
# Rating 1 or 2 -> Negative (0)
# Rating 3 -> Neutral (filtered out for binary classification)
binary_reviews = reviews[reviews['rating'] != 3].copy()
binary_reviews['sentiment'] = (binary_reviews['rating'] >= 4).astype(int)

# ---------------------------------------------------------
# Part 1: Text Cleaning Pipeline
# ---------------------------------------------------------
print("2/5 Cleaning review text (lowercasing, punctuation, stopwords, lemmatization)...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Keep negations as they carry strong sentiment signals
stop_words.discard('not')
stop_words.discard('no')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation and special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # 3. Tokenize by whitespace
    tokens = text.split()
    # 4. Remove stop words & apply lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(cleaned_tokens)

binary_reviews['cleaned_text'] = binary_reviews['review_text'].apply(clean_text)

# Save cleaned dataset
binary_reviews.to_csv("data/processed/cleaned_reviews.csv", index=False)
print("Saved cleaned reviews to data/processed/cleaned_reviews.csv")

# ---------------------------------------------------------
# Part 2: TF-IDF Vectorization
# ---------------------------------------------------------
print("3/5 Building TF-IDF matrix (unigrams & bigrams)...")
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X = tfidf.fit_transform(binary_reviews['cleaned_text'])
y = binary_reviews['sentiment']

# ---------------------------------------------------------
# Part 3: Supervised Sentiment Modeling (Naive Bayes)
# ---------------------------------------------------------
print("4/5 Training Naive Bayes Classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "="*50)
print("SUPERVISED MODEL EVALUATION (NAIVE BAYES):")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Sentiment Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig("docs/sentiment_confusion_matrix.png")
plt.close()
print("Saved confusion matrix plot to docs/sentiment_confusion_matrix.png")

# Analyze Top Positive and Negative Keyword Indicators
feature_names = np.array(tfidf.get_feature_names_out())
# Log-ratio of feature probabilities
log_prob_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
top_neg_idx = np.argsort(log_prob_ratio)[:10]
top_pos_idx = np.argsort(log_prob_ratio)[-10:]

print("\nTop 5 Negative Keywords/Phrases:")
for word in feature_names[top_neg_idx][:5]:
    print(f"  - {word}")

print("\nTop 5 Positive Keywords/Phrases:")
for word in feature_names[top_pos_idx][-5:]:
    print(f"  - {word}")

# ---------------------------------------------------------
# Part 4: VADER Rule-Based Sentiment Analysis & Comparison
# ---------------------------------------------------------
print("\n5/5 Running VADER Rule-Based Benchmark...")
sia = SentimentIntensityAnalyzer()

def get_vader_prediction(text):
    score = sia.polarity_scores(str(text))['compound']
    return 1 if score >= 0.05 else 0

binary_reviews['vader_pred'] = binary_reviews['review_text'].apply(get_vader_prediction)
vader_acc = accuracy_score(binary_reviews['sentiment'], binary_reviews['vader_pred'])
nb_acc = accuracy_score(y_test, y_pred)

print(f"Naive Bayes Accuracy (on test set): {nb_acc:.2%}")
print(f"VADER Lexicon Accuracy (overall):   {vader_acc:.2%}")

# ---------------------------------------------------------
# Part 5: Top Customer Complaints / Themes Visualization
# ---------------------------------------------------------
print("\n5/5 Extracting top complaint themes and generating plot...")

# Filter only negative reviews (rating 1 or 2)
negative_corpus = binary_reviews[binary_reviews['sentiment'] == 0]['cleaned_text']
neg_words = " ".join(negative_corpus).split()

# Remove generic non-informative words if any
custom_filter = {'food', 'order', 'delivery', 'uber', 'eats'}
filtered_neg_words = [w for w in neg_words if w not in custom_filter]

# Get the top 8 most frequent complaint keywords
top_complaints = pd.Series(filtered_neg_words).value_counts().head(8).reset_index()
top_complaints.columns = ['Complaint Theme', 'Frequency']

# Create horizontal bar chart
plt.figure(figsize=(8, 4.5))
sns.barplot(
    data=top_complaints,
    x='Frequency',
    y='Complaint Theme',
    hue='Complaint Theme',
    palette='Reds_r',
    legend=False
)
plt.title('Top Drivers of Customer Dissatisfaction (Complaint Themes)', fontsize=13, weight='bold')
plt.xlabel('Frequency in Negative Reviews', fontsize=11)
plt.ylabel('Complaint Keyword', fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the visualization
chart_path = "docs/top_complaint_themes.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved complaint themes chart to {chart_path}")