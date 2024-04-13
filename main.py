import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Clean and preprocess text
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Set of English stopwords
stopwords_set = set(stopwords.words('english'))

# Preprocessing the dataset to remove punctuation, lowercase, remove stopwords, and apply stemming
corpus = []
for text in df['text']:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

# Vectorization of the corpus using bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the Random Forest classifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluating the classifier
print(clf.score(X_test, y_test))

# Example to classify a new email
email_to_classify = "Example email text to classify."
email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
email_text = ' '.join(email_text)

# Vectorizing the new email and making a prediction
X_email = vectorizer.transform([email_text])
print(clf.predict(X_email))
