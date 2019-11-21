import pandas as pd
import numpy as np
import pickle
import sklearn
import nltk
import re
import warnings  # To ignore warnings
warnings.filterwarnings('ignore')

try:
    data_columns = ["label", "ids", "date", "flag", "user", "tweet"]
    data  = pd.read_csv("data.csv", encoding = "ISO-8859-1", names = data_columns)
except FileNotFoundError:
    print("File does not exist!")


def tokenization(data):
    tokenizer = nltk.tokenize.TweetTokenizer()

    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.remove('no')
    stop_words.remove('not')

    lemma_function = nltk.stem.wordnet.WordNetLemmatizer()

    document = []
    for text in data:
        collection = []
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            if token not in stop_words:
                if '#' in token:
                    collection.append(lemma_function.lemmatize(token))
                else:
                    collection.append(
                        lemma_function.lemmatize(re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", " ", token)))
        document.append(" ".join(collection))
    return document

cleaned_data = tokenization(data.tweet)
data['cleaned_data'] = cleaned_data
train_set, test_set, train_labels, test_labels = sklearn.model_selection.train_test_split(data['cleaned_data'],
                                                                                                    data['label'],
                                                                                                    random_state = 2,
                                                                                                    test_size = 0.2)

cv = sklearn.feature_extraction.text.CountVectorizer()
cv.fit(train_set)

test_features_vectorized = cv.transform(test_set)
test_features_vectorized

train_features_vectorized = cv.transform(train_set)
train_features_vectorized

logistic_model_cv = sklearn.linear_model.LogisticRegression()
logistic_model_cv.fit(train_features_vectorized, train_labels)

pred = logistic_model_cv.predict(test_features_vectorized)

print('Accuracy_score :', sklearn.metrics.accuracy_score(test_labels, pred))

f1_score_logistic_cv = sklearn.metrics.f1_score(test_labels.values, pred, pos_label=0)
print('F1 :', f1_score_logistic_cv)

sklearn.metrics.confusion_matrix(test_labels, pred)

with open('model/logistic_model.pkl', 'wb') as file:
    pickle.dump(cv, file)
    pickle.dump(logistic_model_cv, file)