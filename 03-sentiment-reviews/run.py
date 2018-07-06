from bs4 import BeautifulSoup
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


wordnet_lemmatizer = WordNetLemmatizer()

with open('stopwords.txt') as f:
    stopwords = [word.lower().strip() for word in f]

data = []

with open('electronics/negative.review', mode='rt') as f:
    soup = BeautifulSoup(f, 'lxml')
data += [(el.text.strip(), 0) for el in soup.findAll('review_text')]
with open('electronics/positive.review', mode='rt') as f:
    soup = BeautifulSoup(f, 'lxml')
data += [[el.text.strip(), 1] for el in soup.findAll('review_text')]

data = np.array(data)
X = data[:, 0]
Y = data[:, 1]


def tokenizer(s):
    tokens = nltk.tokenize.word_tokenize(s.lower())
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    return [t for t in tokens if t not in stopwords]

vocabulary = set()

for x in X:
    vocabulary = vocabulary.union(set(tokenizer(x)))

vocabulary = {w: i for i, w in enumerate(vocabulary)}


def tokens_to_vect(tokens, vocabulary):
    x = np.zeros(len(vocabulary))
    for token in tokens:
        x[vocabulary[token]] += 1
    return x/x.sum()


X_tokenized = np.array([tokenizer(x) for x in X])
X_vectorized = np.array([tokens_to_vect(x, vocabulary) for x in X_tokenized])


X_v_train, X_v_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.3)

model = LogisticRegression()
model.fit(X_v_train, Y_train)

score = model.score(X_v_test, Y_test)
print('Classification score: {}'.format(score))

threshold = 0.5
for word in vocabulary:
    w = model.coef_[0][vocabulary[word]]
    if abs(w) > threshold:
        print(word, w)
