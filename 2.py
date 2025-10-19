# -*- coding: utf-8 -*-
"""
Proje: Amazon Müzik Aletleri Yorumu Duygu Analizi
Hedef: Kullanıcı yorumlarını olumlu, olumsuz veya nötr olarak sınıflandırmak.
Kullanılan Yöntemler: TF-IDF, SMOTE, Lojistik Regresyon
"""


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# Download necessary NLTK resources
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

#2: Load Dataset
dataset = pd.read_csv("Instruments_Reviews.csv")
print("Dataset Shape:", dataset.shape) 
print("First 5 Rows:\n", dataset.head()) #ilk 5 row

#3: Handle Missing Values
dataset["reviewText"].fillna("", inplace=True)
dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]
dataset.drop(columns=["reviewText", "summary"], inplace=True) #gereksiz sutunlari kaldir
print("Null Values:\n", dataset.isnull().sum())

#4: Sentiment Labeling
def Labelling(Rows):
    if Rows["overall"] > 3.0:
        return "Positive"
    elif Rows["overall"] < 3.0:
        return "Negative"
    else:
        return "Neutral"

dataset["sentiment"] = dataset.apply(Labelling, axis=1) #sentiment sutunu olustur
print("Sentiment Counts:\n", dataset["sentiment"].value_counts())

#5: Text Preprocessing
Stopwords = set(stopwords.words("english")) - {"not"}
Lemmatizer = WordNetLemmatizer()
Tokenizer = RegexpTokenizer(r'\w+')  # Use RegexpTokenizer to tokenize words without relying on punkt

def Text_Processing(Text):
    Tokens = Tokenizer.tokenize(Text.lower())  # Tokenize using RegexpTokenizer
    Processed_Text = [Lemmatizer.lemmatize(word) for word in Tokens if word not in Stopwords]
    return " ".join(Processed_Text)

dataset["reviews"] = dataset["reviews"].apply(Text_Processing)
print("Processed Dataset:\n", dataset.head())

# Step 6: Visualization
# Ratings Distribution
dataset["overall"].value_counts().plot(kind="pie", autopct="%1.2f%%", figsize=(8, 8))
plt.title("Ratings Distribution")
plt.show()

# Sentiment Distribution
dataset["sentiment"].value_counts().plot(kind="bar", color="skyblue", figsize=(8, 5))
plt.title("Sentiment Distribution")
plt.show()

# Step 7: Feature Engineering
# Encode sentiments
Encoder = LabelEncoder()
dataset["sentiment"] = Encoder.fit_transform(dataset["sentiment"])

# Apply TF-IDF Vectorizer
TF_IDF = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = TF_IDF.fit_transform(dataset["reviews"])
y = dataset["sentiment"]

print("TF-IDF Shape:", X.shape)

# Step 8: Resampling with SMOTE
Balancer = SMOTE(random_state=42)
X_resampled, y_resampled = Balancer.fit_resample(X, y)

print("Original Distribution:", Counter(y))
print("Resampled Distribution:", Counter(y_resampled))

# Step 9: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

# Step 10: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 11: Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
ConfusionMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", ConfusionMatrix)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 12: Plot Confusion Matrix
def plot_cm(cm, classes, title, normalized=False, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalized:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")

plot_cm(ConfusionMatrix, classes=["Positive", "Neutral", "Negative"], title="Confusion Matrix")
plt.show()
