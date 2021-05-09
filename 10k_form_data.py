import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import time

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
#below authenticates using json. Bad practice storing encrypted json on github
creds = ServiceAccountCredentials.from_json_keyfile_name('steps-10000-bc6e9ed43c4c.json', scope) #Change to your downloaded JSON file name
client = gspread.authorize(creds)

#Change to your Google Sheets Name
#can add more spreadsheets as in example - spreadsheets = ['dummy_10k_response','dummy_data_pcr_test']
spreadsheets = ['Results']


def main(spreadsheets):
	df = pd.DataFrame()

	for spreadsheet in spreadsheets:
		# Open the Spreadsheet
		sh = client.open(spreadsheet)

		# Get all values in the first worksheet
		worksheet = sh.get_worksheet(0)
		data = worksheet.get_all_values()

		# Save the data inside the temporary pandas dataframe
		df_temp = pd.DataFrame(columns=[i for i in range(len(data[0]))])
		for i in range(1, len(data)):
			df_temp.loc[len(df_temp)] = data[i]

		#Convert column names
		column_names = data[0]
		df_temp.columns = [convert_column_names(x) for x in column_names]

		# Data Cleaning
		df_temp['Response'] = df_temp['Response'].replace({'': 'Yes'})


		# Concat Dataframe
		df = pd.concat([df, df_temp])

		# API Limit Handling
		time.sleep(5)


	df.to_csv('10k_survey_google_output.csv', index=False)

def convert_column_names(x):
	if x == 'Timestamp':
		return 'date_time'
	elif x == 'What is your name?':
		return 'User'
	elif x == 'I walked 10000 steps today':
		return 'Response'
	else:
		return x


if __name__ == '__main__':
	print('Scraping Form Data')
	main(spreadsheets)

#second stage - read newly outputed file and run data cleaning steps
df = pd.read_csv("10k_survey_google_output.csv", parse_dates=[0])#,index_col=0)


print("Data loaded! starting data cleaning...")

#time zone correction
df['GMT_delta'] = np.where(df['User'] == 'Buskie', 6, 0)#,
                               #np.where(df['User'] == 'Watson', 12,
                                         #np.where(df['User'] == 'Sam H', 1,
                                                  #0)))
df['GMT_delta'] = pd.to_datetime(df.GMT_delta, format='%H') - pd.to_datetime(df.GMT_delta, format='%H').dt.normalize()
df['date_time'] = df['date_time'] - df['GMT_delta']

#convert from datetime to date
df['date_time'] = df['date_time'].dt.date
#remove column
df = df.drop(['GMT_delta'], axis=1)
#remove date duplicates
df = df.drop_duplicates(
    subset = ["date_time", 'User'],
    keep = 'last').reset_index(drop=True)

#replace 'Yes' with 1 int
df['Response'] = df['Response'].replace(['Yes'],'1').astype(str).astype(int)


#calculate metrics
df['steps'] = df.apply(lambda row: row.Response * 10000, axis=1)
#df['distance'] = df.apply(lambda row: row.steps *(11.8/15678), axis=1)

df['relative_dist'] = np.where(df['User'] == 'Darnell', 0.00069, 0.00075)#,
df['distance'] = df['steps'] * df['relative_dist']
df['user_total_distance'] = df.groupby(by=['User'])['distance'].transform(lambda x: x.cumsum())
df = df.drop(['relative_dist'], axis=1)
# here could normalise distance if dan distance is less.

#calculate cumulative response per user and their dominance
df['cum_sum'] = df["Response"].cumsum()
df['cum_steps'] = df['steps'].cumsum()
df['user_cum'] = df.groupby(by=['User'])['Response'].transform(lambda x: x.cumsum())
df['user_cum_steps'] = df.user_cum*10000
df['Current_dominance'] = round(100*df.user_cum/df["Response"].sum(),2)
df['dominance'] = round(100*df.user_cum/df.cum_sum,2)
df['user_location'] = df['User']

#add location coordinates

df_location = pd.DataFrame(
    {'user_location': ['Buskie', 'Darnell', 'Ewan', 'Keith', 'Matthew', 'Rusty', 'Sam H', 'Sam J', 'Stirling', 'Watson'],
     'City': ['Houston', 'Banchory', 'Auchterarder', 'Edinburgh', 'Glasgow', 'Banchory', 'Den Hague', 'Edinburgh', 'Edinburgh', 'Melbourne'],
     'Latitude': [29.788560, 57.053191365939014, 56.297822940458516, 55.96489428639169, 55.614565375840684, 57.059500, 52.01156076443694, 55.973031019654556, 55.97611497379852,-37.80644503373699],
     'Longitude': [-95.404690, -2.494927207289044, -3.700814331824761, -3.193001769495062, -4.497515324553008, -2.470750, 4.3537398921012835, -3.1942029310662634, -3.1693718812876726, 144.96365372001142]})
df = df.merge(df_location, on='user_location', how='left')

#df['user_radius'] = df.groupby(by=['User'])['user_total_distance'].transform(lambda x: x.max()*1000)

#Write df to csv for visualisation
df.to_csv('Response_data_for_visualisation.csv', index=False)
print("Written response data for visualisation, check 'Response_data_for_visualisation.csv'")

#Create dataframe for bar_char_race
df_race = pd.pivot_table(df, index='date_time', columns= 'User', values= "user_cum")#, aggfunc=[np.sum], fill_value=0)
df_race = df_race.fillna(method='ffill')
df_race = df_race.fillna(value=0)
df_race.to_csv('10k_race_data_wide.csv', index='date_time')
print("Written 10k race response data in wide format for bar_chart_race visualisation, check '10k_race_data_wide.csv'")
print(df_race)
print(df_location)


