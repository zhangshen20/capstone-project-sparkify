#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


# import libraries
from pyspark.sql import SparkSession

from pyspark.sql.functions import avg, col, concat, desc, explode,         lit, min, max, sum, count, split, udf, to_date, from_unixtime, datediff, when, countDistinct, date_add

from pyspark.sql import DataFrame
from pyspark.sql import types

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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

