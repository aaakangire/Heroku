# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:27:11 2020

@author: Aashu
"""


from flask import Flask, render_template, request, redirect
import contractions
import re
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

pickle_in = open('clf_knn.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_in = open('Logistic_clf_cv_1.pkl', 'rb')
cv = pickle.load(pickle_in)

def trend(headlines):
    news = ' '.join(str(x) for x in headlines)
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    sm = PorterStemmer()
    corpus = []
    news= re.sub("[^a-zA-Z]", " ", news)
    news = contractions.fix(news)
    news = news.lower()
    news = word_tokenize(news) 
    news = [sm.stem(word) for word in news if not word in set(all_stopwords)]
    news = ' '.join(news)
    corpus.append(news)
    X = cv.transform(corpus)
    index = classifier.predict(X)
    return index

app = Flask(__name__)

headlines = []

@app.route('/')
def hello():
    return "Hello World"

@app.route('/home', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        headline = request.form['headline']
        headlines.append(headline)
        return redirect('/home')
    else: 
        return render_template('index.html')
    
@app.route('/home/post')
def post():
    index = trend(headlines)
    return render_template('index.html', index = index, news = headlines)

@app.route('/home/refresh')
def refresh():
    headlines.clear()
    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug = True)
    
