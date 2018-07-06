import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('spambase.data', header=None).as_matrix()

X, Y = data[:, :-1], data[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Classification score Naive Bayes: {}'.format(model.score(X_test, Y_test)))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Classification score Naive AdaBoost: {}'.format(model.score(X_test, Y_test)))
