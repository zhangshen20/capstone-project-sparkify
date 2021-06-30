from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, sum, count, split, udf, to_date, from_unixtime, datediff, when, countDistinct, date_add
from pyspark.sql import DataFrame
from pyspark.sql import types

import findspark
findspark.init()


def extract_past_N_days(df: DataFrame, \
                        days: types.IntegerType, \
                        date_col_event: types.StringType) -> DataFrame:
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
        
        df_new = \
            df.join(df_max_date, on='userId', how='inner') \
              .where(datediff(col('date_to'), col(date_col_event)) < days) \
              .withColumn('date_from', \
                          when(datediff(col('date_to'), col('registration_date')) < days, col('registration_date')).otherwise(date_add(col('date_to'), -1*days+1)) \
                         )
        
    except Py4JJavaError as e:
        
        df_new = sc.emptyRDD()
        
    return df_new

def remove_invalid_record(df: DataFrame) -> DataFrame:
    ''' remove invalid record (UsedId is Not null)
    
    Parameters:
        df (DataFrame): a dataframe
    Returns:
        df_new: a dataframe contains only valid record
    
    '''
    return df.where((df.userId != "") & (df.userId.isNotNull()))


if __name__ == '__main__':

    # create a Spark session
    spark = SparkSession.builder \
            .master("local") \
            .appName("sparkify_data_engineering") \
            .getOrCreate()

    mini_dataset = "mini_sparkify_event_data.json"
    df = spark.read.json(mini_dataset)
    cnt = df.count()
    print("df.count {}".format(cnt))

    df = remove_invalid_record(df)
    cnt = df.count()
    print("df.count {}".format(cnt))