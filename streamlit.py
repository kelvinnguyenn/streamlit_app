import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from requests_html import HTMLSession


st.title('Internet Outage Liklihood and Severity Prediction')

# Section 1: Dropdown Menus

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


# Section 4: Model Computation


# Section 5: Compare Two Countries



