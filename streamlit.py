import streamlit
import pandas as pd
import numpy as np

# Section 1: Dropdown Menus


# Section 2: Showing the news articles that will be used
# API stuff (News API)


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


# Section 5: Compare Two Countries



