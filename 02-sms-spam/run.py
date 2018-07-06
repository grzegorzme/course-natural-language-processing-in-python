import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def visualize(messages):
    words = ' '.join(msg.lower() for msg in messages)
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    data = pd.read_csv('sms-spam-collection-dataset.zip',
                       compression='zip', encoding='ISO-8859-1').as_matrix()

    X, Y = data[:, 1], data[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    vectorizers = [CountVectorizer, TfidfVectorizer]
    models = [AdaBoostClassifier, RandomForestClassifier, MultinomialNB]

    for vectorizer in vectorizers:
        for model in models:
            v = vectorizer()
            v.fit(X_train)
            X_train_vect = v.transform(X_train)

            m = model()
            m.fit(X_train_vect, Y_train)

            print('vect: {} | model: {} | score: {}'.format(
                v.__class__.__name__,
                m.__class__.__name__,
                m.score(v.transform(X_test), Y_test))
            )

    visualize(X[Y == 'spam'])
    visualize(X[Y == 'ham'])
