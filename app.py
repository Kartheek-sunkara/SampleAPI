from flask import Flask,abort,render_template,request,redirect,url_for,jsonify
import os
import spacy
import tabula
from flashtext import KeywordProcessor
import tika
keyword_processor = KeywordProcessor()
from tika import parser
import xml.etree.ElementTree as ET
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import string
import re
import json
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stopwords = stopwords.words('english')
from sklearn.model_selection import train_test_split
import spacy
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
#import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()    
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def classification():  
    result=request.args['name']
    # df=pd.read_excel("D:\Pubmed-Share\AutoTaskAllocation\MetaData\mechanismDatasetNew.xlsx")
    df=pd.read_excel(result)
    train, test = train_test_split(df, test_size=0.33, random_state=42)
    punctuations = string.punctuation
    
    # Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
    def cleanup_text(docs, logging=False):
        texts = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)
    
        
        
    
    STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    
    class CleanTextTransformer(TransformerMixin):
        def transform(self, X, **transform_params):
            return [cleanText(text) for text in X]
        def fit(self, X, y=None, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}
        
    def cleanText(text):
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.lower()
        return text
    
    def tokenizeText(sample):
        tokens = parser(sample)
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas
        tokens = [tok for tok in tokens if tok not in STOPLIST]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
        return tokens
    
    def printNMostInformative(vectorizer, clf, N):   
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        topClass1 = coefs_with_fns[:N]
        topClass2 = coefs_with_fns[:-(N + 1):-1]
        
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
    clf = LinearSVC()
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    
    # data
    train1 = train['articleText'].tolist()
    labelsTrain1 = train['Mechanism'].tolist()
    
    test1 = test['articleText'].tolist()
    labelsTest1 = test['Mechanism'].tolist()
    # train
    pipe.fit(train1, labelsTrain1)
    
    # test
    preds = pipe.predict(test1)
    
    
    printNMostInformative(vectorizer, clf, 10)
    
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
    transform = pipe.fit_transform(train1, labelsTrain1)
    vocab = vectorizer.get_feature_names()
    
    for i in range(len(train1)):
        s = ""
        indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
        numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
        for idx, num in zip(indexIntoVocab, numOccurences):
            s += str((vocab[idx], num))
    
    df_prediction=pd.DataFrame(columns=['actual', 'pred'])
    df_prediction['actual']=labelsTest1
    df_prediction['pred']= preds
    print(df_prediction)
    
    return "JSONP_data"
    
    
if __name__ == "__main__":
     app.debug = True
     app.run()
