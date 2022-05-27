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
region = st.selectbox(
    'Which region are you interested in?',
    ('Middle East/Northern Africa', 'Southeast Asia'))


with st.form("Country selection"):

    if (region == 'Middle East/Northern Africa'):
        country = st.selectbox(
            'Which country in the {} are you interested in?'.format(region),
            ('Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'))
    else:
        country = st.selectbox(
            'Which country in the {} are you interested in?'.format(region),
            ('Brunei', 'Burma', 'Myanmar', 'Cambodia', 'Timor-Leste', 'Indonesia', 'Laos', 'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Vietnam'))

    submitted = st.form_submit_button("Click to scrape for articles.")

# Section 2: Showing the news articles that will be used
# API stuff (News API)

n_articles = 10

if submitted:
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
    st.write(bodies)




# Section 3: Map Section
# Loads in country data
country_locations = pd.read_csv('https://raw.githubusercontent.com/albertyw/avenews/master/old/data/average-latitude-longitude-countries.csv')

# makes columns numeric so they can be passed in map function
middle_east = ['Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen']
se_asia = ['Brunei', 'Burma', 'Myanmar', 'Cambodia', 'Timor-Leste', 'Indonesia', 'Laos', 'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Vietnam']

ioda = pd.read_csv('ioda.csv')
ioda_filtered = ioda[((ioda['location_name'].isin(middle_east)) | (ioda['location_name'].isin(se_asia))) & (ioda['overlaps_window'] == False)].reset_index().drop(columns=['index'])
ioda_filtered['region'] = ioda_filtered['location_name'].apply(lambda x: 'Southeast Asia' if x in se_asia else 'Middle East')
ioda_filtered = ioda_filtered.sort_values(by='start').reset_index(drop=True)
score_threshold = np.exp(7) - 1
ioda_filtered['severe'] = ioda_filtered['score'] >= score_threshold

print(ioda_filtered.columns)

start_date = str(datetime.datetime.fromtimestamp(min(ioda_filtered['start'])))
latest_date = str(datetime.datetime.fromtimestamp(max(ioda_filtered['start'])))

start_date = start_date[:start_date.find(' ')].split('-')
latest_date = latest_date[:latest_date.find(' ')].split('-')

print(start_date)
# Makes column names lowercase
country_locations.rename(columns = {'Latitude':'latitude', 'Longitude':'longitude', 'Country':'country'}, inplace=True)

# Creates a timeframe scrollbar.

from datetime import datetime


start_time = st.slider(
     "Select time range",
     value=(datetime(int(start_date[0]), int(start_date[1]), int(start_date[2])), datetime(int(latest_date[0]), int(latest_date[1]), int(latest_date[2]))),
     format="MM/DD/YY")

# st.write("Time range chosen: ", start_time)

# Plots only the data that is a country in the Middle East or Southeast Asia
country_locations_filtered = country_locations[(country_locations.country.isin(middle_east)) | (country_locations.country.isin(se_asia))].reset_index()
st.map(country_locations_filtered)

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



