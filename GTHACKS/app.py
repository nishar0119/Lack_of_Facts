import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)
model = pickle.load(open('models.pkl', 'rb'))



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    true_news = pd.read_csv("True.csv")
    fake_news = pd.read_csv("Fake.csv")
    #true_news.shape
    #print(true_news.head())
    true_text = true_news["title"].tolist()
    fake_text = fake_news["title"].tolist()
    all_news = fake_text + true_text
    labels = ([0] * len(fake_text) + ([1] * len(true_text)))
    train_data, test_data, train_labels, test_labels = train_test_split(all_news, labels, test_size = 0.2, random_state = 1)
    counter = CountVectorizer()
    counter.fit(train_data)
    train_counts = counter.transform(train_data)
    test_counts = counter.transform(test_data)
    news_counts = counter.transform(["Macron calls Paris beheading 'Islamist terrorist attack'"])
    #print(train_data[3])
    classifier = MultinomialNB()
    classifier.fit(train_counts, train_labels)
    predictions = classifier.predict(test_counts)
    print(accuracy_score(test_labels, predictions))
    print((classifier.predict(news_counts) ))
    print(classifier.predict_proba(news_counts))
    #pickle.dump(classifier, open('pick.pkl', 'wb'))
    #model = pickle.load(open('pick.pkl', 'rb'))
    
    if request.method == 'POST':
        data1 = request.form['a']
        data = [data1]
        vect = counter.transform(data).toarray()
        my_pred = classifier.predict(vect)
    return render_template('after.html', prediction=my_pred)

if __name__ == "__main__":
   app.run(debug=True)

