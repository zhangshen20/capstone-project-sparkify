from numpy.lib.function_base import _diff_dispatcher
import pandas as pd
import time

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession, column
from pyspark.sql.functions import from_json, avg, col, concat, desc, explode, lit, min, max, sum, count, split, udf, to_date, from_unixtime, datediff, when, countDistinct, date_add
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, FloatType, DateType, TimestampType

from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml import PipelineModel

from pyspark.sql.functions import explode, sum, split, col, expr, when

import requests

from util.utils import add_date_field, Normalize_Data_ToUser, extract_past_N_days, Choose_Features, remove_invalid_record

training_file = 'spark_user_data.csv'


def f_merge_df(microdf, batchid):

    print(f"inside forEachBatch for batchid:{batchid}. Rows in passed dataframe: {microdf.count()}")
    
    if (DeltaTable.isDeltaTable(spark, silver_data_path) == False):

        microdf \
        .write \
        .format("delta") \
        .mode("overwrite") \
        .save(silver_data_path)
        
    else:

        deltadf = DeltaTable.forPath(spark, silver_data_path)
    
        (deltadf.alias("t")
                .merge(
                    microdf.alias("s"),
                    "s.event_date = t.event_date and s.ts = t.ts and s.userId = t.userId")
                .whenNotMatchedInsertAll()
                .execute()
        )

if __name__ == '__main__':

    # df_pd = pd.read_csv(training_file)
    # df_sum = df_pd.sum()

    col_list = ['About', 'Downgrade', 'Cancel']    

    schema = StructType([ 
        StructField("ts",TimestampType(),True), 
        StructField("userId",StringType(),True), 
        StructField("sessionId",IntegerType(),True), 
        StructField("page", StringType(), True),
        StructField("auth", StringType(), True),
        StructField("method", StringType(), True),
        StructField("status", IntegerType(), True),
        StructField("level", StringType(), True),
        StructField("itemInSession", IntegerType(), True),
        StructField("location", StringType(), True),
        StructField("userAgent", StringType(), True),
        StructField("lastName", StringType(), True),
        StructField("firstName", StringType(), True),
        StructField("registration", LongType(), True),
        StructField("gender", StringType(), True),
        StructField("artist", StringType(), True),
        StructField("song", StringType(), True),
        StructField("length", FloatType(), True)
    ])

    schemaDelta = StructType([ 
        StructField("ts",TimestampType(),True), 
        StructField("userId",StringType(),True), 
        StructField("sessionId",IntegerType(),True), 
        StructField("page", StringType(), True),
        StructField("auth", StringType(), True),
        StructField("method", StringType(), True),
        StructField("status", IntegerType(), True),
        StructField("level", StringType(), True),
        StructField("itemInSession", IntegerType(), True),
        StructField("location", StringType(), True),
        StructField("userAgent", StringType(), True),
        StructField("lastName", StringType(), True),
        StructField("firstName", StringType(), True),
        StructField("registration", LongType(), True),
        StructField("gender", StringType(), True),
        StructField("artist", StringType(), True),
        StructField("song", StringType(), True),
        StructField("length", FloatType(), True),
        StructField("registration_date", DateType(), True),
        StructField("days_from_registration", IntegerType(),True),
        StructField("event_date", DateType(), True)
    ])                

#     spark = SparkSession \
#             .builder \
#             .appName("WebApp") \
#             .config("spark.sql.shuffle.partitions", 2) \
#             .getOrCreate()

    spark = SparkSession.builder \
        .appName("WebApp") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()    

    from delta.tables import *

    gbt_model = PipelineModel.load('trained_model/GBTClassificationModel')
#     lr_model = PipelineModel.load('trained_model/LogisticRegressionModel')
#     rf_model = PipelineModel.load('trained_model/RandomForestClassificationModel')

#     df_sparkify = spark.read.csv(training_file, header=True, inferSchema=True)
#     df_sparkify = df_sparkify.persist()

    # unpivotExpr = "stack(3, 'About', About, 'Cancel', Cancel, 'Downgrade', Downgrade) as (Page, Cnt)"
    # df_main = df_sparkify \
    #             .select(col('About'), col('Downgrade'), col('Cancel')) \
    #             .agg(sum('About').alias('About'), sum('Downgrade').alias('Downgrade'), sum('Cancel').alias('Cancel')) \
    #             .select(expr(unpivotExpr))

    # df_main = df_main.persist()

    base_path = "path"

    raw_data_path = f"{base_path}/raw/destination/"
    raw_data_ckpt = f"{base_path}/raw/checkpoint/"

    bronze_data_path = f"{base_path}/bronze/destination/"
    bronze_data_ckpt = f"{base_path}/bronze/checkpoint/"    

    silver_data_path = f"{base_path}/silver/destination/"
    silver_data_ckpt = f"{base_path}/silver/checkpoint/"

    # Step 1: Receive data via socket and injest into as Text format.
    # Base on experiment, TEXT consumes less memory

    queryRaw = spark \
            .readStream \
            .format("socket") \
            .option("host", "localhost") \
            .option("port", 3001) \
            .load() \
            .writeStream \
            .outputMode("append") \
            .format("text") \
            .option("checkpointLocation", raw_data_ckpt) \
            .option("path", raw_data_path) \
            .trigger(processingTime='60 seconds') \
            .start()                

    # Step 2: Convert text into delta format

    df_text = spark \
            .readStream \
            .format("text") \
            .load(raw_data_path) \

    df_text = df_text.withColumn("jsonValue", from_json(df_text.value, schema)) \
                 .select(col("jsonValue.ts").alias('ts'), 
                         col("jsonValue.userId").alias('userId'),
                         col("jsonValue.sessionId").alias('sessionId'),
                         col("jsonValue.page").alias('page'),
                         col("jsonValue.auth").alias('auth'),
                         col("jsonValue.method").alias('method'),
                         col("jsonValue.status").alias('status'),
                         col("jsonValue.level").alias('level'),
                         col("jsonValue.itemInSession").alias('itemInSession'),
                         col("jsonValue.location").alias('location'),
                         col("jsonValue.userAgent").alias('userAgent'),
                         col("jsonValue.lastName").alias('lastName'),
                         col("jsonValue.firstName").alias('firstName'),
                         col("jsonValue.registration").alias('registration'),
                         col("jsonValue.gender").alias('gender'),
                         col("jsonValue.artist").alias('artist'),
                         col("jsonValue.song").alias('song'),
                         col("jsonValue.length").alias('length')
                         )

    df_text = add_date_field(remove_invalid_record(df_text))

    # df_raw = df_raw.withWatermark("ts", "10 hours")

    queryBronze = df_text \
            .writeStream \
            .partitionBy("event_date") \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", bronze_data_ckpt) \
            .option("path", bronze_data_path) \
            .trigger(processingTime='120 seconds') \
            .start()    

    time.sleep(180)

    querySilver = spark \
            .readStream \
            .format("delta") \
            .load(bronze_data_path) \
            .writeStream \
            .outputMode("append") \
            .format("delta") \
            .foreachBatch(f_merge_df) \
            .option("checkpointLocation", silver_data_ckpt) \
            .trigger(processingTime='120 seconds') \
            .start()

            # .option("path", delta_data_path) \



    # spark.sql("create table events USING DELTA LOCATION 'path/delta/'")

    # df_action = spark \
    #         .readStream \
    #         .format("parquet") \
    #         .option("path", raw_data_path) \
    #         .schema(schema) \
    #         .load()

    # df_action = spark.read.parquet(raw_data_path)
    # N_days = 3000
    # df_action = add_date_field(df_action)
    # df_action = Normalize_Data_ToUser(extract_past_N_days(df_action, days=N_days, date_col_event='event_date'))
    # df_action.show()

    # df_max_date = spark \
    #         .read \
    #         .format("parquet") \
    #         .option("path", raw_data_path) \
    #         .schema(schema) # \
    #         # .load()
    # df_max_date = add_date_field(df_max_date)
    # df_max_date = df_max_date.groupBy('userId').agg(max('event_date').alias('date_to'))

    # N_days = 3000
    # # df_action = extract_past_N_days(df_action, days=N_days, date_col_event='event_date')
    # df_test = df_action.groupBy(["userId"]).agg(max('event_date').alias('date_to'))

    # queryTest = df_test \
    #         .writeStream \
    #         .queryName("dfTest") \
    #         .outputMode("complete") \
    #         .format("memory") \
    #         .start()         

    # N_days = 3000

    # # userCounts = df_action.groupBy(["userId", "page"]).count()
    # # df_action = Normalize_Data_ToUser(extract_past_N_days(df_action, days=N_days, date_col_event='event_date'))

    # df_action = extract_past_N_days(df_action, days=N_days, date_col_event='event_date')

    # query = df_action \
    #         .writeStream \
    #         .queryName("UserQuery") \
    #         .outputMode("append") \
    #         .format("memory") \
    #         .trigger(processingTime='20 seconds') \
    #         .start()  

    # wordCounts = words.groupBy("Page").count()

    # df_max_date = df_action.groupBy('userId').agg(max('event_date').alias('date_to'))

    # queryMaxDate = df_max_date \
    #         .writeStream \
    #         .queryName("dfMaxDate") \
    #         .outputMode("complete") \
    #         .format("memory") \
    #         .start()

    # queryAction = df_action \
    #         .writeStream \
    #         .queryName("dfAction") \
    #         .outputMode("append") \
    #         .format("memory") \
    #         .start()            

    while(1==1):

        print(">>>>>-----------------------------------------------------------------------<<<<<")
        print("queryRaw Id: {}".format(queryRaw.id))
        print("queryRaw RunId: {}".format(queryRaw.runId))
        print("queryRaw Name: {}".format(queryRaw.name))
        # print("queryRaw Explain: {}".format(queryRaw.explain()))

        print("---------------------------------------------------------------------------------")
        print("queryBronze Id: {}".format(queryBronze.id))
        print("queryBronze RunId: {}".format(queryBronze.runId))
        print("queryBronze Name: {}".format(queryBronze.name))
        # print("queryBronze Explain: {}".format(queryBronze.explain()))        

        print("---------------------------------------------------------------------------------")
        print("querySilver Id: {}".format(querySilver.id))
        print("querySilver RunId: {}".format(querySilver.runId))
        print("querySilver Name: {}".format(querySilver.name))
        # print("querySilver Explain: {}".format(querySilver.explain()))        


        # df_delta = spark \
        #         .read \
        #         .format("parquet") \
        #         .load(raw_data_path)
        # df_delta.show()                

        # **************************************************************************************************
        # Note: If no data is ready in 'delta_data_path', you will get error by running below statement.
        #       It will be good to let the streaming process run for a bit longer
        # **************************************************************************************************
        df_action = spark \
                    .read \
                    .format("delta") \
                    .load(silver_data_path)

        cnt = df_action.count()

        print(' --- Number of Records: {}'.format(cnt))

        N_days = 3000
        # df_action = add_date_field(df_action)
        df_action = Normalize_Data_ToUser(extract_past_N_days(df_action, days=N_days, date_col_event='event_date'))
        # df_action.show(20)
        # # df_action = df_action.withColumn('label', df.churn)
        cols = ['num_songs', \
                'num_artist', \
                'num_active_days', \
                'Add Friend', \
                'Add to Playlist', \
                'Downgrade', \
                'Thumbs Up', \
                'days_since_registration', \
                'avg_songs_per_active_day', \
                'avg_play_length_per_active_day']

        df_action = Choose_Features(df_action, cols)

        gbt_predictions = gbt_model.transform(df_action)
        gbt_df_out = gbt_predictions.select('userId', 'rawPrediction', 'prediction', 'probability')
        # gbt_df_out.show(40)
        gbt_df_out = gbt_df_out.groupBy('prediction').count()

        # # df_stream = spark.sql(
        # #     """select   page
        # #             ,   sum(count) as Cnt 
        # #         from    WordCount 
        # #         group by page
        # #     """)
        # # df_stream.show()

        # # df_update = df_main \
        # #             .join(df_stream, on="Page", how="left") \
        # #             .select(df_main.Page, (df_main.Cnt + when(df_stream.Cnt.isNull(), 0).otherwise(df_stream.Cnt)).alias('count')) \

        # # df_update.show()

        # # df_stream = spark.sql("select * from dfTest")

        # # df_update = df_sparkify \
        # #             .join(df_stream, on="userId", how="left") \
        # #             .select("*")

        # # df_update.show()        

        # top_tags = list(gbt_df_out.select(col('prediction')).toPandas()['prediction'])
        # # # extract the counts from dataframe and convert them into array
        # tags_count = list(gbt_df_out.select(col('count')).toPandas()['count'])
        # # initialize and send the data through REST API
        # url = 'http://localhost:5001/updateData'
        # request_data = {'label': str(top_tags).replace('bytearray', ''), 'data': str(tags_count)}
        # response = requests.post(url, data=request_data)

        # # send_df_to_dashboard(hashtag_counts_df)
        # # spark.sql("select word, count from counting").show()
        time.sleep(10)