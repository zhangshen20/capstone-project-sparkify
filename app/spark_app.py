import pandas as pd
import time

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession, column
from pyspark.sql.functions import col

from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml import PipelineModel

from pyspark.sql.functions import explode, sum, split, col, expr, when

import requests

training_file = 'spark_user_data.csv'

if __name__ == '__main__':

    # df_pd = pd.read_csv(training_file)
    # df_sum = df_pd.sum()

    col_list = ['About', 'Downgrade', 'Cancel']    

    spark = SparkSession \
            .builder \
            .appName("WebApp") \
            .config("spark.sql.shuffle.partitions", 2) \
            .getOrCreate()

    df_sparkify = spark.read.csv(training_file, header=True, inferSchema=True)         

    unpivotExpr = "stack(3, 'About', About, 'Cancel', Cancel, 'Downgrade', Downgrade) as (Page, Cnt)"

    df_main = df_sparkify \
                .select(col('About'), col('Downgrade'), col('Cancel')) \
                .agg(sum('About').alias('About'), sum('Downgrade').alias('Downgrade'), sum('Cancel').alias('Cancel')) \
                .select(expr(unpivotExpr))

    df_main = df_main.persist()

    lines = spark \
            .readStream \
            .format("socket") \
            .option("host", "localhost") \
            .option("port", 3001) \
            .load()

    words = lines.select(
        explode(
            split(lines.value, " ")
        ).alias("Page")
    )

    wordCounts = words.groupBy("Page").count()

    query = wordCounts \
            .writeStream \
            .queryName("PageCount") \
            .outputMode("complete") \
            .format("memory") \
            .start()    

    while(1==1):

        df_stream = spark.sql("select Page, count as Cnt from PageCount")
        df_update = df_main \
                    .join(df_stream, on="Page", how="left") \
                    .select(df_main.Page, (df_main.Cnt + when(df_stream.Cnt.isNull(), 0).otherwise(df_stream.Cnt)).alias('count')) \

        df_update.show()

        top_tags = list(df_update.select(col('Page')).toPandas()['Page'])
        # # extract the counts from dataframe and convert them into array
        tags_count = list(df_update.select(col('count')).toPandas()['count'])
        # initialize and send the data through REST API
        url = 'http://localhost:5001/updateData'
        request_data = {'label': str(top_tags).replace('bytearray', ''), 'data': str(tags_count)}
        response = requests.post(url, data=request_data)

        # send_df_to_dashboard(hashtag_counts_df)
        # spark.sql("select word, count from counting").show()
        time.sleep(10)


    # df = spark.read.csv(training_file, header=True, inferSchema=True)
    # df = df.withColumn('label', df.churn)       

    # gbt_model = PipelineModel.load('trained_model/GBTClassificationModel')
    # lr_model  = PipelineModel.load('trained_model/LogisticRegressionModel')
    # rf_model  = PipelineModel.load('trained_model/RandomForestClassificationModel')


    # splits = df.randomSplit([0.9, 0.1], seed=42)

    # df_train = splits[0]
    # df_test  = splits[1]        

    # gbt_predictions = gbt_model.transform(df_test)
    # lr_predictions  = lr_model.transform(df_test)
    # rf_predictions  = rf_model.transform(df_test)

    # gbt_df_out = gbt_predictions.select('userId', col('prediction').alias('gbt_pred'), 'churn')
    # lr_df_out  = lr_predictions.select('userId', col('prediction').alias('lr_pred'))
    # rf_df_out  = rf_predictions.select('userId', col('prediction').alias('rf_pred'))

    # df_out = gbt_df_out.join(lr_df_out, on='userId', how='inner') \
    #                    .join(rf_df_out, on='userId', how='inner') \
    #                    .select(gbt_df_out.userId, gbt_df_out.gbt_pred, lr_df_out.lr_pred, rf_df_out.rf_pred, gbt_df_out.churn).toPandas()

    # gbt_evaluator = BinaryClassificationEvaluator()
    # gbt_result = gbt_evaluator.evaluate(gbt_predictions)

    # lr_evaluator = BinaryClassificationEvaluator()
    # lr_result = lr_evaluator.evaluate(lr_predictions)

    # rf_evaluator = BinaryClassificationEvaluator()
    # rf_result = rf_evaluator.evaluate(rf_predictions)    