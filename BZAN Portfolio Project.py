#!/usr/bin/env python
# coding: utf-8

# # Predictive Modeling of NFL Play by Play Trends
The Objective of the project is to uses NFL pbp data to develop a time series forecating model that accurately predicts
future trends in performance.

Diving into:
- Time Series
- ARIMA Model
- FB Prophet Model
# ### Importing libraries

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima, ARIMA, model_selection
warnings.filterwarnings('ignore')
from prophet import Prophet


# ### Load and concat DF

# In[2]:


# all pbp dada from 2014 - 2024

df1 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2014.csv")
df2 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2015.csv")
df3 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2016.csv")
df4 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2017.csv")
df5 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2018.csv")
df6 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2019.csv")
df7 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2020.csv")
df8 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2021.csv")
df9 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2022.csv")
df10 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2023.csv")
df11 = pd.read_csv(r"D:\temp Portfolio Project\pbp-2024.csv")
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11], axis=0)


# ### Preprocessing and describing

# In[3]:


# dropping empty columns

df.dropna(how='all', axis=1, inplace=True)
df.head()


# In[148]:


df.info()


# In[5]:


df.duplicated()
df.shape


# In[6]:


# Creating a column to show field goal instances
df['IsFieldGoal'] = df['Description'].str.contains('FIELD GOAL IS GOOD', case=False, na=False).astype(int)
df['IsExtraPoint'] = df['Description'].str.contains('EXTRA POINT IS GOOD', case=False, na=False).astype(int)
df['IsSafety'] = df['Description'].str.contains(', SAFETY', case=False, na=False).astype(int)

# Verifying values are int
df['IsFieldGoal'] = df['IsFieldGoal'].astype('int64')
df['IsExtraPoint'] = df['IsExtraPoint'].astype('int64')
df['IsSafety'] = df['IsSafety'].astype('int64')


# In[73]:


# Counting the total Points scored on a given play

df['y'] = (
    df['IsTouchdown'] * 6 +
    df['IsFieldGoal'] * 3 +
    df['IsExtraPoint'] * 1 +
    df['IsSafety'] * 2 +
    df['IsTwoPointConversionSuccessful'] * 2
)


# In[8]:


df['GameDate'] = pd.to_datetime(df['GameDate'], format='%Y-%m-%d')
#df['GameDate'] = pd.to_datetime(df['GameDate'], errors='coerce')
#df.index = pd.to_datetime(df.index, errors='coerce', format='%Y-%m-%d')


# In[9]:


df.set_index('GameDate', inplace=True)


# In[10]:


df.head()


# ### Forecasting using Points Scored

# In[11]:


# First, make sure your index is datetime
df.index = pd.to_datetime(df.index)

# Group by day (you could change to W for week, M for month if you want)
daily_points = df['y'].resample('D').sum()

# Now plot the aggregated daily scores
plt.figure(figsize=(20, 5))
plt.plot(daily_points.index, daily_points, label='Total Points per Day')

# Set clean x-ticks: just label the full dates naturally
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # one tick per year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # format ticks as years only
plt.xticks(rotation=90)

plt.xlabel('Date')
plt.ylabel('Points')
plt.title('Total Points Scored Per Day Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


# perform seasonal decomposition
# y represents points scored

result = seasonal_decompose(df['y'], model='additive', period=4)


# In[13]:


# plotting the components in the graph

sns.set(style='whitegrid')

plt.figure(figsize=(18,12))

# trend component
plt.subplot(411)
sns.lineplot(data=result.trend)
plt.title('Trend')
plt.xticks(rotation=90)

# seasonal component
plt.subplot(412)
sns.lineplot(data=result.seasonal)
plt.title('Seasonal')
plt.xticks(rotation=90)

# Residuals component
plt.subplot(413)
sns.lineplot(data=result.resid)
plt.title('Residuals')
plt.xticks(rotation=90)

# Original data
plt.subplot(414)
sns.lineplot(data=df['y'])
plt.title('Original Data')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### Indexing data by specfic game date

# In[14]:


# Play by play for the month of September 2024

df.loc['2024-09']


# ### Adding in more variables for future references

# In[17]:


# Creating timestamp to signify the start of week

nfl_week1_starts = {
    2014: pd.Timestamp('2014-09-04'),
    2015: pd.Timestamp('2015-09-10'),
    2016: pd.Timestamp('2016-09-08'),
    2017: pd.Timestamp('2017-09-07'),
    2018: pd.Timestamp('2018-09-06'),
    2019: pd.Timestamp('2019-09-05'),
    2020: pd.Timestamp('2020-09-10'),
    2021: pd.Timestamp('2021-09-09'),
    2022: pd.Timestamp('2022-09-08'),
    2023: pd.Timestamp('2023-09-07'),
    2024: pd.Timestamp('2024-09-05')
}


# In[18]:


# Mapping each Year to its Week 1 Start Date

df['Week1Start'] = df['SeasonYear'].map(nfl_week1_starts)


# In[19]:


# Calculating Season Week

df['SeasonWeek'] = ((df.index - df['Week1Start']).dt.days // 7) + 1


# In[20]:


# Dummy variables to add to distinguish between regular season and playoffs, and season week

df['IsRegular'] = (
    ((df['SeasonYear'] <= 2020) & (df['SeasonWeek'] <= 17)) |   # 17-week seasons
    ((df['SeasonYear'] >= 2021) & (df['SeasonWeek'] <= 18))     # 18-week seasons
).astype(int)


# In[21]:


# Keeping only rows after Week 1

df = df[df['SeasonWeek'] >= 1]


# In[22]:


# Preview

print(df[['SeasonYear', 'SeasonWeek']].head())


# In[23]:


# df does not include data on super bowl

df.loc['2024-02']


# In[25]:


# Daily points df without the missing days where a game did not occur

daily_points_only = daily_points[daily_points != 0]
daily_points_only.head()


# In[26]:


# Testing rolling points scored

#df['3_game_rolling'] = df['y'].rolling(window = 21).mean()


# ### ARIMA Model

# In[74]:


# weekly dataframe for modeling

weekly_stats = df.resample('W').agg({
    'y': 'sum',
    'Down': 'mean',
    'ToGo': 'mean',
    'SeriesFirstDown': 'sum',
    'Yards': 'sum',
    'IsRush': 'sum',
    'IsPass': 'sum',
    'IsTouchdown': 'sum',
    'IsFieldGoal': 'sum',
    'IsSack': 'sum',
    'IsInterception': 'sum',
    'IsPenalty': 'sum',
    'PenaltyYards': 'mean'
})
weekly_stats.head()


# In[75]:


# daily stats for additional modeling

daily_stats = df.resample('D').agg({
    'y': 'sum',
    'Down': 'mean',
    'ToGo': 'mean',
    'SeriesFirstDown': 'sum',
    'Yards': 'sum',
    'IsRush': 'sum',
    'IsPass': 'sum',
    'IsTouchdown': 'sum',
    'IsFieldGoal': 'sum',
    'IsSack': 'sum',
    'IsInterception': 'sum',
    'IsPenalty': 'sum',
    'PenaltyYards': 'mean'
})


# In[76]:


# Plotting the partial autocorrelation (PACF) of weekly points scored

fig, ax = plt.subplots(figsize = (10,6))
plot_pacf(weekly_stats['y'], lags = 100, ax = ax)
plt.show()


# In[77]:


# Plotting the autocorrelation (ACF) of weekly points scored

fig, ax = plt.subplots(figsize = (10,6))
plot_acf(weekly_stats['y'], lags = 100, ax = ax)
plt.show()


# In[78]:


# train and testing data for weekly data

train = weekly_stats.loc[:'2024-01']
test = weekly_stats.loc['2024-08':]


# In[79]:


#plotting train and test so we can visually see the split
train = weekly_stats.loc[:'2024-01']
test = weekly_stats.loc['2024-08':]

# plotting train and test so we can visually see the split
test[['y']]     .rename(columns={'y': 'TEST DATA'})     .join(
        train[['y']].rename(columns={'y': 'TRAIN DATA'}),
        how='outer'
    ) \
    .plot(figsize=(15,5), title='Train/Test Split of Points Scored', style='.')
plt.show()


# In[81]:


# Using pmdarima for the arima model and the best parameters
model = auto_arima(train['y'],
                  seasonal = False)
model.summary()


# In[82]:


# Predictions 
predictions_arima = model.predict(n_periods = len(test))
predictions_arima


# In[83]:


# Model assesment of weekly stats
model_assessment(train['y'], test['y'], predictions_arima, "ARIMA")


# In[ ]:


# Due to the nature of an NFL season and how there are only so many months that games occur in
# the ARIMA does not quite accurately predict actual seasonal values. Lets try the FB Prophet Model and
# see if its abled to pick up on the seasonal NFLschededule trends


# ### FP Prophet Model

# In[133]:


# Splitting the data for FB Prophet modeling 
pro_train = weekly_stats.loc[:'2024-01']
pro_test = weekly_stats.loc['2024-08':]


# In[134]:


weekly_stats_train = pro_train.reset_index()     .rename(columns={'GameDate': 'ds'})


# In[135]:



get_ipython().run_cell_magic('time', '', "model = Prophet(yearly_seasonality=True, # NFL season cycle September–February\n    weekly_seasonality=True,             # Weekly NFL games (Sunday, MNF, TNF)\n    daily_seasonality=False,             # You DON'T need daily seasonality for weekly points\n    seasonality_mode='additive',         # Additive usually fits scoring patterns better (linear growth)\n    changepoint_prior_scale=0.1,         # Moderate flexibility to capture trend changes (injuries, playoffs)\n    interval_width=0.8,                  # 80% prediction intervals for uncertainty\n)\nmodel.fit(weekly_stats_train)")


# In[136]:


weekly_stats_test = pro_test.reset_index()     .rename(columns={'GameDate': 'ds'})

weekly_stats_test_frct = model.predict(weekly_stats_test)


# In[137]:


weekly_stats_test_frct.head()


# In[138]:


# Actual vs. forecasted values from your model 

fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(weekly_stats_test_frct, ax=ax)
plt.show()


# In[139]:


# Trends, seasonality, and residuals

fig  = model.plot_components(weekly_stats_test_frct)
plt.show()


# ### Answering how well the model performed

# In[145]:


# Reset the correct index
weekly_stats_test = weekly_stats_test.set_index('ds')
weekly_stats_test.index = pd.to_datetime(weekly_stats_test.index)

# Plotting the forecast with the actual
fig, ax = plt.subplots(figsize=(15, 5))
fig = model.plot(weekly_stats_test_frct, ax=ax)
ax.scatter(weekly_stats_test.index, weekly_stats_test['y'], color='red', label='Actual')
ax.set_title('Forecast vs Actual', fontsize=16)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
plt.show()

