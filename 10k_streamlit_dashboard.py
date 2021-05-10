import pandas as pd
import streamlit as st
import numpy as np
import pydeck as pdk
import datetime
import bar_chart_race as bcr
import math
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue

import time
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


st.title('10k Challenge')
#st.image('./RDGSC2.png', caption='Get out in those hills and get your steps')
st.image('./RDGSC3.png', caption='Get out into those hills and get your steps')
#st.image('./RDGSC.jpg')

#from PIL import Image
#image = Image.open('sunrise.jpg')
#st.image(image, caption='Get out in those hills and get your steps')


#caching the data if def load_data is reading and manipulating large data source
#@st.cache
def load_data(nrows):
    data = pd.read_csv('Response_data_for_visualisation.csv')#, nrows=nrows)
    data['date_time'] = pd.to_datetime(data['date_time'])
    return data

#@st.cache
def load_race_data(nrows):
    data = pd.read_csv('10k_race_data_wide.csv')#, nrows=nrows)
    data['date_time'] = pd.to_datetime(data['date_time'])
    return data

def load_raw_data(nrows):
    data = pd.read_csv('10k_survey_google_output.csv')
    data['date_time'] = pd.to_datetime(data['date_time'])
    return data

#call the functions
raw_data = load_raw_data(10000)
clean_data = load_data(10000)
race = load_race_data(10000).set_index('date_time') #date_time set as index in race dataframe for bar_chart_race

#create dirty double table
dirty_doubles = pd.DataFrame(raw_data['User'].value_counts())
dirty_doubles['dubious_response'] = raw_data['User'].value_counts()
dirty_doubles['clean_response'] = clean_data['User'].value_counts()
dirty_doubles['number of dirty doubles'] = dirty_doubles['User']-dirty_doubles['clean_response']
dirty_doubles = dirty_doubles.sort_values(by=['number of dirty doubles'], ascending=False)
print(dirty_doubles[['dubious_response', 'clean_response', 'number of dirty doubles']])


#most popular days
pop_days = pd.DataFrame(clean_data['date_time'])
pop_days['response_date'] = pd.to_datetime(pop_days['date_time'])
pop_days['day'] = pop_days['response_date'].dt.day_name()
print(pop_days)
#can this be ranked by date - monday to sunday for histogram? also make it bespoke for each person?
num_days = pop_days['day'].value_counts()
print(num_days)




#https://stackoverflow.com/questions/54694957/pandas-average-row-count-per-day-of-the-week
# Create series for days in your dataframe
#days_in_df = df['day'].value_counts()

# Create a dataframe with all days
#start = '01/01/2019'
#end = '01/31/2019'
#all_days_df = pd.DataFrame(data={'datetime': pd.date_range(start='01/01/2019', periods=31, freq='d')})
#all_days_df['all_days'] = all_days_df['datetime'].dt.day_name()

# Use that for value counts
#all_days_count = all_days_df['all_days'].value_counts()

# We now merge them
#result = pd.concat([all_days_count, days_in_df], axis=1, sort=True)

# Finnaly we can get the ration
#result['day'] / result['all_days']


leaderboard = pd.DataFrame(clean_data['User'].value_counts())
leaderboard['Rank'] = leaderboard['User'].rank(method='min', ascending=False)
leaderboard['Total Entires'] = clean_data['User'].value_counts()
#leaderboard = leaderboard.sort_values(by=['User'], ascending=False)

st.subheader("Leaderboard")
st.write(leaderboard[['Rank', 'Total Entires']])

st.subheader("The most popular day for walking is...")
st.write(num_days)

st.subheader("Do y'all wanna see the data?")
if st.checkbox('yeah, show me the data!'):
    st.subheader('Your wish is my command')
    st.write(clean_data[['date_time', 'User', 'steps', 'user_cum', 'dominance', 'user_cum_steps', 'user_total_distance', 'cum_sum', 'cum_steps']])

#check box for race data
#st.subheader('Wanna see the race data too?')
#if st.checkbox("yeah, don't hold back"):
#    st.subheader('Alrighty then')
#    st.write(race)

if st.checkbox("I didn't ask you to hold back, show me the dirty doubles too!"):
    st.subheader('If you must')
    st.write(dirty_doubles[['dubious_response', 'clean_response', 'number of dirty doubles']])

st.subheader('Need to see the raw data too?')
if st.checkbox("yeah, I said don't hold back"):
    st.subheader('Alrighty then')
    st.write(raw_data)

#Bar chart
#st.bar_chart(raw_data['user_cum'])

#histogram
#df = pd.DataFrame(raw_data[:200], columns = ['user_cum','user_cum_steps'])
#df.hist()
#plt.show()
#st.pyplot()

#line chart
#st.line_chart(clean_data)

#map
# Use pandas to calculate additional data
df_clean = pd.DataFrame(clean_data)
#df_clean["user_radius"] = df_clean.groupby(by=['User'])["user_total_distance"].transform(lambda x: x.max())
print(raw_data)
print(clean_data)
print(race)

#TODO import datetime, link slider to radius.
#sldier for date selection on map
st.subheader("Date Range")
x = st.slider('Choose a date within the 10k challenge',
              min_value=datetime.date(2021,2,1), max_value=datetime.date(2021,11,1))
st.write("Date:", x)

#create dumb map - max distance walked. could improve with slider vs time to show progress.
df_location = pd.DataFrame(
    {'User': ['Ali', 'Buskie', 'Darnell', 'Ewan', 'Keith', 'Matthew', 'Rusty', 'Sam H', 'Sam J', 'Stirling', 'Watson'],
     'City': ['Aberdeen', 'Houston', 'Banchory', 'Auchterarder', 'Edinburgh', 'Glasgow', 'Banchory', 'Den Hague', 'Edinburgh', 'Edinburgh', 'Melbourne'],
     'Latitude': [57.12664782485791, 29.788560, 57.053191365939014, 56.297822940458516, 55.96489428639169, 55.614565375840684, 57.059500, 52.01156076443694, 55.973031019654556, 55.97611497379852,-37.80644503373699],
     'Longitude': [-2.1194205303315305, -95.404690, -2.494927207289044, -3.700814331824761, -3.193001769495062, -4.497515324553008, -2.470750, 4.3537398921012835, -3.1942029310662634, -3.1693718812876726, 144.96365372001142],
     'user_radius': [1.0, 1.0, 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0]})
grouped = pd.DataFrame(
{#'User': ['Buskie', 'Darnell', 'Ewan', 'Keith', 'Matthew', 'Rusty', 'Sam H', 'Sam J', 'Stirling', 'Watson'],
 'user_radius': df_clean.groupby("User")['user_total_distance'].max()})
print(grouped)
df_location = df_location.merge(grouped, on='User', how='left') # this doesn't work since no 'user_radius' column in df_location.
df_location['distance_walked (km)'] = round(df_location.user_radius_y/1,0)
#grouped = df_clean.groupby("User")['user_total_distance'].max()
#print(grouped)
#none of these below give the right max total distance by user. Need to fix this, then fix the scale so it does not scoll with the map.
#df_location["user_radius"] = clean_data.groupby('User')["user_radius"].transform(lambda x: x.max()) # this only works when 'user_radius' is calculated as the max of user total distance in 10k_form_data
#df_location['user_radius'] = clean_data.groupby('User')['user_total_distance'].transform('max')
#df_location["user_radius3"] = clean_data.groupby(by=['User'])['user_total_distance'].transform(lambda x: x.max()*1000)
print(df_location)


# Define a layer to display on a map
st.subheader('Where has all this walking taken us?')
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=57.051536150778986,
        longitude=-2.5052866770165534,
        zoom=4,
        bearing=0,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=df_location,
            pickable=True,
            opacity=0.05,
            stroked=True,
            filled=True,
            radius_scale=1000, #convert from m to km,
            #radius_min_pixels=1,
            #radius_max_pixels=100,
            line_width_min_pixels=1,
            get_position=['Longitude', 'Latitude'],
            get_radius="user_radius_y",
            radiusUnits="meters",
            get_fill_color=[255, 140, 0], # can pass a function here to change color based on position
            get_line_color=[0, 0, 0],
        ),
    ],
    tooltip={"text": "{User}\nhas walked\n{distance_walked (km)}km"}
))

#https://deckgl.readthedocs.io/en/latest/layer.html
# Set the viewport location
#view_state = pdk.ViewState(latitude=57.051536150778986, longitude=-2.5052866770165534, zoom=10, bearing=0, pitch=0)

# Render
#r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{User}\n{user_total_distance}"})
#r.to_html("scatterplot_layer.html")




print(race)
#####RACE########

st.subheader("This isn't a race, but if it was, it would probably be the best race in the world")


with st.spinner(text='video loading, please remain calm'):
    time.sleep(35)
st.success('almost there... almost there... almost there...')

bcr.bar_chart_race(
    df=race,
    filename='10k_race_video.mp4',
    orientation='h',
#    sort='desc',
#    n_bars=10,
#    fixed_order=False,
#    fixed_max=True,
#    steps_per_period=10,
#    interpolate_period=False,
#    label_bars=True,
#    bar_size=.95,
#    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    period_fmt='%B %d, %Y',
#   period_summary_func=lambda v, r: {'x': .99, 'y': .18,
#                                      's': f'Total deaths: {v.nlargest(6).sum():,.0f}',
#                                      'ha': 'right', 'size': 8, 'family': 'Courier New'},
#    perpendicular_bar_func='median',
#    period_length=500,
#    figsize=(5, 3),
#    dpi=144,
#    cmap='dark12',
    title='10k raffle ticket race')#,
#    title_size='',
#    bar_label_size=7,
#    tick_label_size=7,
#    shared_fontdict={'family' : 'Helvetica', 'color' : '.1'},
#    scale='linear',
#    writer=None,
#    fig=None,
#    bar_kwargs={'alpha': .7},
#    filter_column_colors=False)

st.video('./10k_race_video.mp4')
#automation chron works, but is unix based and mac must be on. anachron can do it if mac is offline, but has been depreciated and replaced by launchd (https://medium.com/swlh/how-to-use-launchd-to-run-services-in-macos-b972ed1e352)
#https://www.jcchouinard.com/python-automation-with-cron-on-mac/
#0 0 * * * cd /Users/stephenbuskie/PycharmProjects/10k && /Users/stephenbuskie/opt/anaconda3/envs/streamlit/bin/python 10k_form_data.py