from pyspark.sql import SparkSession, column
from util.utils import add_date_field, Normalize_Data_ToUser, extract_past_N_days, Choose_Features, remove_invalid_record
from util.config import *
import quinn

def spaces_to_underscores(s):
    return s.replace(" ", "_")

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("TransformSparkifyUser") \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()    

    from delta.tables import *    

    if (DeltaTable.isDeltaTable(spark, silver_data_path)):

        df_action = spark \
                    .read \
                    .format("delta") \
                    .load(silver_data_path)
        N_days = 3000
        df_action = Normalize_Data_ToUser(extract_past_N_days(df_action, days=N_days, date_col_event='event_date'))

        df_action = quinn.with_columns_renamed(spaces_to_underscores)(df_action)
        # df_action.show()

        df_action.write.format("delta").mode("overwrite").save(gold_data_path)