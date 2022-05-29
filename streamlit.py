import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from requests_html import HTMLSession
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# Section 1: Dropdown Menus

st.title('Internet Outage Liklihood and Severity Prediction')

st.write('Select first Country')
with st.form("First Country selection"):
    region = st.selectbox(
        'Which region are you interested in?',
        ('Middle East/Northern Africa', 'Southeast Asia'))
    if (region == 'Middle East/Northern Africa'):
        country = st.selectbox(
            'Which country in the {} are you interested in?'.format(region),
            ('Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'))
    else:
        country = st.selectbox(
            'Which country in the {} are you interested in?'.format(region),
            ('Brunei', 'Burma', 'Myanmar', 'Cambodia', 'Timor-Leste', 'Indonesia', 'Laos', 'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Vietnam'))
    submitted1 = st.form_submit_button("Click to select 2nd country.")
st.write('Select Second Country')
with st.form("Second Country selection"):
    region = st.selectbox(
        'Which region are you interested in?',
        ('Middle East/Northern Africa', 'Southeast Asia'))
    if (region == 'Middle East/Northern Africa'):
        country2 = st.selectbox(
            'Which country in the {} are you interested in?'.format(region),
            ('Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'))
    else:
        country2 = st.selectbox(
            'Which country in the {} are you interested in?'.format(region),
            ('Brunei', 'Burma', 'Myanmar', 'Cambodia', 'Timor-Leste', 'Indonesia', 'Laos', 'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Vietnam'))
    submitted = st.form_submit_button("Click to scrape for articles.")


# Section 2: Showing the news articles that will be used
# API stuff (News API)

n_articles = 10

def getArticle(country):
    # api key for newsapi
    api_key = ''

    # gets the most far back date we can lookup using newsapi
    search_from_date = (datetime.datetime.now() - datetime.timedelta(days = 29)).strftime('%Y-%m-%d')

    url = ('https://newsapi.org/v2/everything?'
        'language=en&'
        'from={}&'
        'pagesize=10&'
        'page=1&'
        'q={}&'
        'searchIn=title&'
        'sortBy=publishedAt&'
        'apiKey={}'.format(search_from_date, country, api_key))

    response = requests.get(url)
    d = response.json() # makes a dictionary object from the response json

    # getting the titles, source, dates, and links of articles
    titles = [article['title'] for article in d['articles']]
    sources = [article['source']['name'] for article in d['articles']]
    publish_date = [article['publishedAt'][:10] for article in d['articles']]
    urls = [article['url'] for article in d['articles']]

    # replacing curly quote "’“”" characters
    titles = [article['title'].replace('’', '\'').replace("“", '"').replace("”", '"') for article in d['articles']]

    st.subheader('Recent Articles for {}'.format(country))

    # displaying the ten most recent articles using a dataframe
    df = pd.DataFrame(list(zip(titles, publish_date, sources, list(range(1, len(titles) + 2)))), columns = ['Title', 'Pub Date', 'Source', 'Index']).set_index('Index')
    # dropping duplicate articles
    df = df.drop_duplicates(subset = ['Title'])
    st.table(df[:10])

    # scraping article bodies
    bodies = []

    for url in urls:
        print('getting {}'.format(url))
        session = HTMLSession()
        elements = session.get(url).html.find('p')
        bodies.append(' '.join([element.text for element in elements]))
    #st.write(bodies)
    return bodies




# Section 3: Map Section


# Section 4: Model Computation
loaded_vectorizer = pickle.load(open('data/vectorizer.sav', 'rb'))
loaded_classifier = pickle.load(open('data/classifier.sav', 'rb'))
loaded_regression = pickle.load(open('data/regression.sav', 'rb'))

def classification(articles):
    matrix = loaded_vectorizer.transform(articles)
    return loaded_classifier.predict(matrix)

def regression(articles):
    matrix = loaded_vectorizer.transform(articles)
    return np.exp(loaded_regression.predict(matrix) - 1)
# Section 5: Compare Two Countries

def procCountry(bodies):
    a = classification(bodies)
    r = regression(bodies)
    if a.sum() != 0:
        # take avg for which classifier is True
        mean1=(r*a).sum()/a.sum() 
        # rounding to 2 decimals
        mean1=np.round(mean1, 2)
        return mean1


if submitted:
    bodies=getArticle(country)
    m1=procCountry(bodies)
    bodies=getArticle(country2)
    m2=procCountry(bodies)
    st.write(f'country: {country} Mean : {m1}')
    st.write(f'country: {country2} Mean : {m2}')

