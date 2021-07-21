base_path = "path"

raw_data_path = f"{base_path}/raw/destination/"
raw_data_ckpt = f"{base_path}/raw/checkpoint/"

bronze_data_path = f"{base_path}/bronze/destination/"
bronze_data_ckpt = f"{base_path}/bronze/checkpoint/"    

silver_data_path = f"{base_path}/silver/destination/"
silver_data_ckpt = f"{base_path}/silver/checkpoint/"

gold_data_path = f"{base_path}/gold/destination/"
gold_data_ckpt = f"{base_path}/gold/checkpoint/"

shuffle_partitions = 4

trigger_seconds = 2
trigger_time = f"{trigger_seconds} seconds"