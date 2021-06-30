#!/usr/bin/env python
# coding: utf-8

# # Sparkify Project Workspace - Part 1
# This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.
# 
# You can follow the steps below to guide your data analysis and model building portion of this project.

# In[1]:


import findspark
findspark.init()


# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, sum, count, split, udf, to_date, from_unixtime, datediff, when, countDistinct, date_add
from pyspark.sql import DataFrame
from pyspark.sql import types

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# create a Spark session
spark = SparkSession.builder  \
                    .master("local") \
                    .appName("sparkify_data_engineering")  \
                    .getOrCreate()

# # Load and Clean Dataset
# In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 

mini_dataset = "mini_sparkify_event_data.json"

df = spark.read.json(mini_dataset)
# df.persist()


# # Exploratory Data Analysis
# When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.
# 
# ### Define Churn
# 
# Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.
# 
# ### Explore Data
# Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

# List attributes of data points
df.printSchema()

# Convert to Pandas Dataframe for initial analysis on smaller dataset
df_pd = df.toPandas()

# In[8]:


# List counts of each attributes

df_pd.count()


# In[9]:


df_pd.head(10)


# In[10]:


# Let's focus on 'Page' attributes

# List the counts of each value on 'page'


group_data = df_pd.groupby(['page'])
bar_names = group_data.count().index
bar_values = group_data.count()['userId']
y_pos = range(len(bar_names))

plt.figure(figsize=(30, 5))

ax1 = plt.subplot(131)
ax1.title.set_text('Page Count')
plt.bar(bar_names, bar_values, 0.6)
plt.xticks(y_pos, bar_names, rotation=90)

group_data = df_pd[df_pd['page'] != 'NextSong'].groupby(['page'])
bar_names = group_data.count().index
bar_values = group_data.count()['userId']
y_pos = range(len(bar_names))

ax2 = plt.subplot(132)
ax2.title.set_text('Page Count excluding "NextSong"')
plt.bar(bar_names, bar_values, 0.6)
plt.xticks(y_pos, bar_names, rotation=90)

plt.show()


# #### Count each value on 'page'
# 
# From bar chart 'Page Count', 'NextSong' is most used actions, which are far more than others.
# 
# After excluding 'NextSong' from dataset, 'Home', 'Thumbs Up', 'Add to Playlist' are three most of actions.

# In[11]:


# data clean
# check if user id is None
# Number of records have no 'userId'

df.where((df.userId == "") | (df.userId.isNull())).count()


# In[12]:


# Remove records which have no 'userId'

df = df.where((df.userId != "") & (df.userId.isNotNull()))


# In[13]:


# Get list of user ids which have churn out.

df_churn = df.where(col('page') == 'Cancellation Confirmation').select(col('userId').alias('userId'), lit(1).alias('churn')).distinct()


# In[14]:


# Number of total users from the dataset.

total_users = df.select(col('userId')).distinct().count()
churn_users = df_churn.count()

print("Total users: {}".format(total_users))
print("Churn users: {}".format(churn_users))
print("Churn rate:  {}".format(churn_users / total_users))


# # Data and Feature Engineering
# Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
# - Write a script to extract the necessary features from the smaller subset of data
# - Ensure that your script is scalable, using the best practices discussed in Lesson 3
# - Try your script on the full data set, debugging your script if necessary
# 
# If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

# ### ####
# Metrics by User
# 
# - number of songs played
# - number of artist played
# - number of active days
# - number of active days for past N days before downgrade
# - number of active days for past N days before cancelled
# - number of active days since registration
# - number of days being 'Paid' member
# - number of days being 'Free' member
# - average songs played per active day
# - total seconds of played 
# 
# ### ####

# In[15]:


# Convert timestamp to Date
# Create columns -> 'registration_datetime' and 'event_datetime', 'days_from_registration'

df = df.withColumn('event_date', from_unixtime(col('ts').cast('bigint') / 1000, 'yyyy-MM-dd'))        .withColumn('registration_date', from_unixtime(col('registration').cast('bigint') / 1000, 'yyyy-MM-dd'))        .withColumn('days_from_registration', datediff(col('event_date'), col('registration_date'))) 


# In[16]:


def extract_past_N_days(df: DataFrame,                         days: types.IntegerType,                         date_col_event: types.StringType) -> DataFrame:
    ''' Extract past N days data from a dataframe
    
    Parameters:
        df (DataFrame): a dataframe
        days (Integer): a number of days
        date_col_event (String): date column name used to check if a record is within the date range
    Returns:
        df_new: a dataframe contains only specitifed days' data
    
    '''
    try:
        df_max_date = df.groupBy('userId').agg(max(date_col_event).alias('date_to'))
        
        df_new =             df.join(df_max_date, on='userId', how='inner')               .where(datediff(col('date_to'), col(date_col_event)) < days)               .withColumn('date_from',                           when(datediff(col('date_to'), col('registration_date')) < days, col('registration_date')).otherwise(date_add(col('date_to'), -1*days+1))                          )
        
    except Py4JJavaError as e:
        
        df_new = sc.emptyRDD()
        
    return df_new


# In[17]:


def Normalize_Data_ToUser(df: DataFrame) -> DataFrame:
    '''Normalize Raw dataframe to User Based. After this transformation, each record will be usedId based. 
    
    Parameters:
        df (DataFrame): raw data frame
        
    Returns:
        df_new (DataFrame): normalized dataframe (user-based)
    '''

    df_userId = df.groupBy('userId')

    df_days =         df_userId.agg(countDistinct('event_date').alias('num_active_days'),                       min('date_from').alias('date_from'),                                        max('date_to').alias('date_to'),                                            min('registration_date').alias('registration_date')                         )
    df_current_level =         df.where(col('event_date') == col('date_to'))           .groupBy('userId')           .agg(min('level').alias('current_level'))           .withColumn('current_level_paid', when(col('current_level') == 'paid', 1).otherwise(0))
    
    df_pages =         df_userId.pivot("page")            .agg(count('page'))    
    
    df_songs =         df.where(col('page') == 'NextSong')           .groupBy('userId')           .agg(count('page').alias('num_songs'), countDistinct('song').alias('num_songs_unique'))

    df_artists =         df.where(col('page') == 'NextSong')           .groupBy('userId')           .agg(countDistinct('artist').alias('num_artist'),                sum('length').alias('total_play_length')               )    

    df_active_days_as_paid =         df.where(col('level') == 'paid').groupBy('userId')           .agg(countDistinct('event_date').alias('num_active_days_paid'))

    df_new =         df_songs.join(df_artists, on='userId', how='left')                 .join(df_days, on='userId', how='left')                 .join(df_active_days_as_paid, on='userId', how='left')                 .join(df_pages, on='userId', how='left')                 .join(df_current_level, on='userId', how='left')                 .select(col('*'),                         (datediff(col('date_to'), col('registration_date'))+1).alias('days_since_registration'),                         (datediff(col('date_to'), col('date_from'))+1).alias('days_in_member') )                 .withColumn('active_pct', (col('num_active_days') / col('days_in_member')))                 .withColumn('avg_songs_per_day', col('num_songs') / col('days_in_member'))                 .withColumn('avg_songs_per_active_day', col('num_songs') / col('num_active_days'))                 .withColumn('avg_play_length_per_day', col('total_play_length') / col('days_in_member'))                 .withColumn('avg_play_length_per_active_day', col('total_play_length') / col('num_active_days')) 

    return df_new


# In[18]:


def Apply_Churn_Flag(df: DataFrame, df_churn: DataFrame) -> DataFrame:
    '''Add 'Churn' flag to df
    
    Parameters:
        df (DataFrame): normalized dataframe
        df_churn (DataFrame): list of churn users 
        
    Return:
        df_new (DataFrame): new dataframe with a churn flag value -> 1
    '''
    
    df_new = df.join(df_churn, on='userId', how='left')                .withColumn('churn', when(col('churn') == 1, 1).otherwise(0))
    
    return df_new
    


# In[19]:


N_days=3000 # Assume we extract all records from the exisintg smaller dataset.

df_N_days = Apply_Churn_Flag(Normalize_Data_ToUser(extract_past_N_days(df, days=N_days, date_col_event='event_date')), df_churn)


# In[20]:


df_N_pd = df_N_days.toPandas()


# In[21]:


df_N_pd.describe()


# In[22]:


# Create a dataframe for the users who have churned out
df_N_pd_churn = df_N_pd[df_N_pd['churn'] == 1]

# Create a dataframe for the users who are still in member
df_N_pd_not_churn = df_N_pd[df_N_pd['churn'] == 0]


# ##### Find the difference between members and users who have churned out

# In[56]:


(df_N_pd_not_churn.describe() - df_N_pd_churn.describe()).transpose()


# #### List of attributes I'd like to further analysis
# 
# - num_songs
# - num_artist
# - num_active_days
# - Add Friend
# - Add to Playlist
# - Downgrade
# - Thumbs Up
# - days_since_registration
# - avg_songs_per_active_day
# - avg_play_length_per_active_day
# 

# In[79]:


cols = ['num_songs',         'num_artist',         'num_active_days',         'Add Friend',         'Add to Playlist',         'Downgrade',         'Thumbs Up',         'days_since_registration',         'avg_songs_per_active_day',         'avg_play_length_per_active_day']


# In[25]:


def Choose_Features(df: DataFrame, cols: list) -> DataFrame:
    '''Choose Features from Dataframe
    
    Parameters:
        df (DataFrame): normalized dataframe
        cols (list): list of columns to be extracted
        
    Return:
        df_new (DataFrame): new dataframe with selected features
    '''

    cols = ['userId', 'churn'] + cols           

    df_new = df.select(cols).na.fill(0)
    
    return df_new
    


# In[88]:


def Choose_Features_Pandas(df_pd, cols):
    '''Choose Features from Dataframe
    
    Parameters:
        df (Pandas DataFrame): normalized dataframe
        cols (list): list of columns to be extracted
        
    Return:
        df_new (DataFrame): new dataframe with selected features
    '''

    cols = ['userId', 'churn'] + cols           

    df_new = df_pd[cols].fillna(0)
    
    return df_new


# In[26]:


import pandas as pd
from IPython.display import display
pd.options.display.max_columns = None

N_days=3000 # Assume we extract all records from the exisintg smaller dataset.

df_N_days = Choose_Features(Apply_Churn_Flag(Normalize_Data_ToUser(extract_past_N_days(df, days=N_days, date_col_event='event_date')), df_churn), cols)


# In[27]:


df_N_pd = df_N_days.toPandas()


# In[53]:


df_N_pd.head(10)


# # Same Methods, Just Work on Big Dataset

# In[57]:


real_dataset = "../sparkify_event_data.json"

df_real = spark.read.json(real_dataset)
df_real.persist()


# In[59]:


# Remove records which have no 'userId'

df_real = df_real.where((df_real.userId != "") & (df_real.userId.isNotNull()))


# In[61]:


# Get list of user ids which have churn out.

df_real_churn = df_real.where(col('page') == 'Cancellation Confirmation').select(col('userId').alias('userId'), lit(1).alias('churn')).distinct()


# In[60]:


df_real = df_real.withColumn('event_date', from_unixtime(col('ts').cast('bigint') / 1000, 'yyyy-MM-dd'))        .withColumn('registration_date', from_unixtime(col('registration').cast('bigint') / 1000, 'yyyy-MM-dd'))        .withColumn('days_from_registration', datediff(col('event_date'), col('registration_date'))) 


# In[62]:


N_days=3000 # Assume we extract all records from the exisintg smaller dataset.

df_real_N_days = Apply_Churn_Flag(Normalize_Data_ToUser(extract_past_N_days(df_real, days=N_days, date_col_event='event_date')), df_real_churn)


# In[63]:


df_real_N_pd = df_real_N_days.toPandas()


# In[64]:


df_real_N_pd.describe()


# In[66]:


# Create a dataframe for the users who have churned out
df_real_N_pd_churn = df_real_N_pd[df_real_N_pd['churn'] == 1]

# Create a dataframe for the users who are still in member
df_real_N_pd_not_churn = df_real_N_pd[df_real_N_pd['churn'] == 0]


# In[68]:


(df_real_N_pd_not_churn.describe() - df_real_N_pd_churn.describe()).transpose()


# In[89]:


df_real_N_days_pd = Choose_Features_Pandas(df_real_N_pd, cols)


# In[91]:


df_real_N_days_pd.head(20)


# ## Write User Level Data Into CSV File for Futhre Use
# 
# To avoid running Spark Jobs multiple times, I write aggregated user-level data into CSV for modelling work.

# In[93]:


df_real_N_pd.fillna(0).to_csv('spark_user_data.csv', index=False)


# # Modelling Work - Refer to 'sparkify_modelling' notebook

# In[ ]:




