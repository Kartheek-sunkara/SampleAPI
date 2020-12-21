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
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stopwords = stopwords.words('english')
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/classification',methods = ['POST', 'GET'])
def classification():  
    #result=request.args.get('pdf_path')
    return "Done"
    
    
if __name__ == "__main__":
     app.debug = True
     app.run()
