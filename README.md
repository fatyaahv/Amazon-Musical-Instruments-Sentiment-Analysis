🎵 Amazon Musical Instruments Review Sentiment Analysis

This project performs sentiment analysis on Amazon Musical Instruments product reviews.
Using Natural Language Processing (NLP) and Machine Learning, it classifies user reviews as Positive, Negative, or Neutral based on their content and star ratings.

The workflow includes data preprocessing, text vectorization (TF-IDF), handling class imbalance using SMOTE, and training a Logistic Regression classifier.

🧭 Project Overview

Goal:
To analyze user feedback on Amazon’s musical instruments and automatically classify the sentiment of each review.

Approach:

Preprocess raw text (cleaning, tokenization, lemmatization, stopword removal).

Convert text into numerical features using TF-IDF.

Handle imbalanced data using SMOTE (Synthetic Minority Oversampling Technique).

Train a Logistic Regression model to predict sentiment.

Evaluate performance using metrics and visualizations.

🧰 Technologies Used
Library	Purpose
Pandas, NumPy	Data manipulation and numerical operations
Matplotlib	Data visualization
NLTK	Text preprocessing (tokenization, stopword removal, lemmatization)
Scikit-learn (sklearn)	Machine learning model building, evaluation, and TF-IDF
Imbalanced-learn (imblearn)	Data balancing using SMOTE
Logistic Regression	Classification algorithm
Python 3.8+	Programming language
📂 Project Structure
📁 Amazon_Sentiment_Analysis/
│
├── Instruments_Reviews.csv     # Dataset file (Amazon musical instruments reviews)
├── sentiment_analysis.py       # Main Python script
└── README.md                   # Project documentation

⚙️ Step-by-Step Workflow
🧩 Step 1: Import Libraries

All required libraries for text processing, visualization, and machine learning are imported.

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

📊 Step 2: Load Dataset

The dataset Instruments_Reviews.csv is loaded using pandas.

dataset = pd.read_csv("Instruments_Reviews.csv")


Displays dataset shape and first 5 rows.

Example columns: reviewText, summary, overall (star rating).

🧼 Step 3: Handle Missing Values

Missing text entries are filled with empty strings.
The columns reviewText and summary are combined into a single column reviews.

dataset["reviewText"].fillna("", inplace=True)
dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]
dataset.drop(columns=["reviewText", "summary"], inplace=True)

💬 Step 4: Sentiment Labeling

Each review is labeled based on its star rating (overall):

Rating > 3.0 → "Positive"

Rating = 3.0 → "Neutral"

Rating < 3.0 → "Negative"

def Labelling(Rows):
    if Rows["overall"] > 3.0:
        return "Positive"
    elif Rows["overall"] < 3.0:
        return "Negative"
    else:
        return "Neutral"

🧹 Step 5: Text Preprocessing

Performed using NLTK tools:

Convert text to lowercase

Tokenize with RegexpTokenizer (ignores punctuation)

Remove English stopwords (except “not” to preserve negations)

Lemmatize words using WordNetLemmatizer

Stopwords = set(stopwords.words("english")) - {"not"}
Tokenizer = RegexpTokenizer(r'\w+')
Lemmatizer = WordNetLemmatizer()

def Text_Processing(Text):
    Tokens = Tokenizer.tokenize(Text.lower())
    Processed_Text = [Lemmatizer.lemmatize(word) for word in Tokens if word not in Stopwords]
    return " ".join(Processed_Text)


This step ensures clean, normalized text ready for feature extraction.

📈 Step 6: Data Visualization

Two plots are generated:

Ratings Distribution (Pie Chart) – visualizes product ratings.

Sentiment Distribution (Bar Chart) – shows how many reviews fall under each sentiment class.

🔠 Step 7: Feature Engineering

Convert text data into numerical format using TF-IDF (Term Frequency–Inverse Document Frequency):

TF_IDF = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = TF_IDF.fit_transform(dataset["reviews"])
y = dataset["sentiment"]


max_features=5000 limits vocabulary size.

ngram_range=(1,2) includes unigrams and bigrams.

⚖️ Step 8: Handling Class Imbalance with SMOTE

SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the dataset by generating synthetic samples of minority classes.

Balancer = SMOTE(random_state=42)
X_resampled, y_resampled = Balancer.fit_resample(X, y)


This ensures that the model doesn’t become biased toward dominant classes.

🧠 Step 9: Model Training (Logistic Regression)

The dataset is split into training (75%) and testing (25%) sets.

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


Logistic Regression is used as a baseline classifier because it performs well on text-based classification problems with TF-IDF features.

📊 Step 10: Model Evaluation

Model predictions are compared to true labels using several metrics:

y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)


Outputs include:

Accuracy Score

Confusion Matrix

Precision, Recall, F1-score per class

A confusion matrix plot is generated using matplotlib.

📉 Example Results
Metric	Value
Accuracy	~0.88 (example)
Positive Class F1	0.91
Neutral Class F1	0.74
Negative Class F1	0.85

(Values may vary depending on dataset and preprocessing.)

🎨 Visualization Example

Confusion Matrix Output:

[[520  45  32]
 [ 60 400  50]
 [ 40  30 560]]


(Each row represents true labels, and columns represent predicted labels.)

🧾 Key Takeaways

TF-IDF + Logistic Regression works well for simple sentiment classification tasks.

SMOTE improves performance by handling unbalanced datasets.

Text preprocessing (especially lemmatization and stopword control) significantly enhances model accuracy.

The model can be further improved by trying SVM, Naive Bayes, or deep learning models (BERT).


👩‍💻 Author

Fatima Humbatli
📍 GitHub: fatyaahv

💬 For questions or contributions, feel free to open an issue.
