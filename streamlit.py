# Import packages
import streamlit
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Section 1: Dropdown Menus


# Section 2: Showing the news articles that will be used
# API stuff (News API)


# Section 3: Map Section


# Section 4: Model Computation
loaded_vectorizer = pickle.load(open('data/vectorizer.sav', 'rb'))
loaded_classifier = pickle.load(open('data/classifier' 'rb'))
loaded_regression = pickle.load(open('data/regression.sav', 'rb'))

def classification(articles):
    matrix = loaded_vectorizer.transform(articles)
    return loaded_classifier.predict(matrix)

def regression(articles):
    matrix = loaded_vectorizer.transform(articles)
    return np.exp(loaded_regression.predict(matrix) - 1)
# Section 5: Compare Two Countries



