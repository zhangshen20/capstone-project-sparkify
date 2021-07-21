import time
import os
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, avg, col, concat, desc, explode, lit, min, max, sum, count, split, udf, to_date, from_unixtime, datediff, when, countDistinct, date_add
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, FloatType, DateType, TimestampType
from pyspark.sql.functions import explode, sum, split, col, expr, when
from pyspark.sql.utils import ForeachBatchFunction

from util.utils import add_date_field, Normalize_Data_ToUser, extract_past_N_days, Choose_Features, remove_invalid_record, stop_stream_query
from util.config import *

def f_merge_df(microdf, batchid):

    # print(f"inside forEachBatch for batchid:{batchid}. Rows in passed dataframe: {microdf.count()}")

    microdf = microdf.dropDuplicates(['ts', 'userId'])
    
    if (DeltaTable.isDeltaTable(spark, silver_data_path) == False):

        microdf \
        .write \
        .partitionBy(["event_year", "event_month", "event_day"]) \
        .format("delta") \
        .mode("overwrite") \
        .save(silver_data_path)
        
    else:

        deltadf = DeltaTable.forPath(spark, silver_data_path)
    
        (deltadf.alias("t")
                .merge(
                    microdf.alias("s"),
                    """ s.event_year  = t.event_year
                    and s.event_month = t.event_month
                    and s.event_day   = t.event_day
                    and s.ts = t.ts 
                    and s.userId = t.userId
                    """)
                .whenNotMatchedInsertAll()
                .execute()
        )

if __name__ == '__main__':

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

    spark = SparkSession.builder \
        .appName(__file__) \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()            

    spark.sparkContext.setLogLevel("ERROR")

    from delta.tables import *

    # Step 1: Receive data via socket and injest into as Text format.
    # Base on experiment, TEXT consumes less memory

    queryRaw = spark \
            .readStream \
            .format("socket") \
            .option("host", "localhost") \
            .option("port", 3001) \
            .load() \
            .writeStream \
            .queryName("queryRaw") \
            .outputMode("append") \
            .format("text") \
            .option("path", raw_data_path) \
            .option("checkpointLocation", raw_data_ckpt) \
            .trigger(processingTime=trigger_time) \
            .start()                

    # Step 2: Convert text into delta format
    df_text = spark \
            .readStream \
            .format("text") \
            .load(raw_data_path)

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

    queryBronze = df_text \
            .writeStream \
            .queryName("queryBronze") \
            .partitionBy(["event_year", "event_month", "event_day"]) \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", bronze_data_ckpt) \
            .option("path", bronze_data_path) \
            .trigger(processingTime=trigger_time) \
            .start()    

    ReadyForNext = False
    while(not ReadyForNext):

        if (os.path.exists(bronze_data_path) 
            and 
            DeltaTable.isDeltaTable(spark, bronze_data_path)):
            ReadyForNext = True
        else:
            time.sleep(10)

    querySilver = spark \
            .readStream \
            .format("delta") \
            .load(bronze_data_path) \
            .writeStream \
            .queryName("querySilver") \
            .outputMode("append") \
            .format("delta") \
            .foreachBatch(f_merge_df) \
            .option("checkpointLocation", silver_data_ckpt) \
            .trigger(processingTime=trigger_time) \
            .start()

    while spark.streams.active != []:
        
        for stream_query in spark.streams.active:
            stop_stream_query(stream_query)

        time.sleep(trigger_seconds)

    exit(0)
