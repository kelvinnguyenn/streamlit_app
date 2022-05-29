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

mena_countries = ('Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen')
sea_countries = ('Brunei', 'Burma', 'Myanmar', 'Cambodia', 'Timor-Leste', 'Indonesia', 'Laos', 'Malaysia', 'Philippines', 'Singapore', 'Thailand', 'Vietnam')


# Model Computation
loaded_vectorizer = pickle.load(open('data/vectorizer.sav', 'rb'))
loaded_classifier = pickle.load(open('data/classifier.sav', 'rb'))
loaded_regression = pickle.load(open('data/regression.sav', 'rb'))

def classification(articles):
    matrix = loaded_vectorizer.transform(articles)
    return loaded_classifier.predict(matrix)

def regression(articles):
    matrix = loaded_vectorizer.transform(articles)
    return np.exp(loaded_regression.predict(matrix) - 1)


# Section 1: Dropdown Menus

st.title('Internet Outage Liklihood and Severity Prediction')

st.write('Select First Country')
with st.form("First Country selection"):
    region = st.selectbox(
        'Which region are you interested in?',
        ('Middle East/Northern Africa', 'Southeast Asia'))
    if (region == 'Middle East/Northern Africa'):
        country = st.selectbox(
            'Which country in the {} are you interested in?'.format(region), mena_countries)
    else:
        country = st.selectbox(
            'Which country in the {} are you interested in?'.format(region), sea_countries)
    submitted = st.form_submit_button("Click to scrape articles for {}.".format(country))



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
# Loads in country data
country_locations = pd.read_csv('https://raw.githubusercontent.com/albertyw/avenews/master/old/data/average-latitude-longitude-countries.csv')

# makes columns numeric so they can be passed in map function
middle_east = list(mena_countries)
se_asia = list(sea_countries)

# Filters only those outages that are equal to or higher than threshold.
ioda = pd.read_csv('ioda.csv')
ioda_filtered = ioda[((ioda['location_name'].isin(middle_east)) | (ioda['location_name'].isin(se_asia))) & (ioda['overlaps_window'] == False)].reset_index().drop(columns=['index'])
ioda_filtered['region'] = ioda_filtered['location_name'].apply(lambda x: 'Southeast Asia' if x in se_asia else 'Middle East')
ioda_filtered = ioda_filtered.sort_values(by='start').reset_index(drop=True)
score_threshold = np.exp(7) - 1
ioda_filtered['severe'] = ioda_filtered['score'] >= score_threshold


# Bounds for timeframe slider. 
start_date = str(datetime.datetime.fromtimestamp(min(ioda_filtered['start'])))
latest_date = str(datetime.datetime.fromtimestamp(max(ioda_filtered['start'])))

start_date = start_date[:start_date.find(' ')].split('-')
latest_date = latest_date[:latest_date.find(' ')].split('-')

# Rename columns
country_locations.rename(columns = {'Latitude':'latitude', 'Longitude':'longitude', 'Country':'country'}, inplace=True)

# Creates a timeframe scrollbar.
from datetime import datetime
import time
start_time = st.slider(
     "Select time range",
     value=(datetime(int(start_date[0]), int(start_date[1]), int(start_date[2])), datetime(int(latest_date[0]), int(latest_date[1]), int(latest_date[2]))),
     format="MM/DD/YY")
st.write("Time range chosen: " + start_time[0].strftime("%m/%d/%Y") + " - " + start_time[1].strftime("%m/%d/%Y"))

# Collects number of outages occurring in the time frame specified by user
display_filter = ioda_filtered[(ioda_filtered['start'] >= start_time[0].timestamp()) & (ioda_filtered['start'] <= start_time[1].timestamp())].groupby('location_name')
display_filter = display_filter.count().rename(columns={'datasource': 'num_outages'}).reset_index()

# Displays number of outages in time frame for each country

# Plots only the data that is a country in the Middle East or Southeast Asia
country_locations_filtered = country_locations[(country_locations.country.isin(middle_east)) | (country_locations.country.isin(se_asia))].reset_index()

# Gets the coordinates for only the countries that have an outage in the time frame.
df = country_locations_filtered[country_locations_filtered.country.isin(display_filter['location_name'])]
df.insert(4,'num_outages', display_filter['num_outages'], True)
df = df.drop(columns=["index"])
df = df.sort_values(by='num_outages', ascending=False)
df['num_outages'] = df['num_outages'].dropna().apply(lambda outages: int(outages))
st.table(df.set_index('country')['num_outages'].dropna())

from urllib.request import urlopen

with urlopen('https://datahub.io/core/geo-countries/r/countries.geojson') as response:
    geo = json.load(response)

# Creates a scatterplot layer with the radius indicating the number of outages
fig = go.Figure(
    go.Choroplethmapbox(
        geojson = geo,
        locations = df.country,
        featureidkey = "properties.ADMIN",
        z = df.num_outages,
        zmin = 0,
        zmax = 750,
        colorscale="sunsetdark",
        marker_opacity=0.5,
        marker_line_width=0,
        visible= True
    )
)
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=1.5,
    mapbox_center = {'lat': -4, 'lon': 70.5},
    width=800,
    height=600,
)


fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig)


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

st.write('Select Second Country')
with st.form("Second Country selection"):
    region = st.selectbox(
        'Which region are you interested in?',
        ('Middle East/Northern Africa', 'Southeast Asia'))
    if (region == 'Middle East/Northern Africa'):
        country2 = st.selectbox(
            'Which country in the {} are you interested in?'.format(region), mena_countries)
    else:
        country2 = st.selectbox(
            'Which country in the {} are you interested in?'.format(region), sea_countries)
    submitted_2nd = st.form_submit_button("Click to scrape for articles.")