# capstone-project-sparkify

### Summary of the Project:

This project is achiving the following goals
  - Transform sparkify member activity data from Udacity S3 location (12GB sized file) into user level data. 
  - Analysis member activities between churned members and active members
  - Create prediction models to help identify potential churned-out members.

### Data Analysis and Modelling (Jupyter Notebook):

- sparkify_data_engineering.ipynb: manual work on cleaning and transforming sparkify raw data

- sparkify_modelling.ipynb: manual work on creating prediction models 


### Project Result

After implementing prediction models, we test it by using 20% of data. The accuracy is 73.9% from Gradient-Boosted Tree Model.

### Web Application Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - Go to 'app' folder ('cd app')
    - To run data enginnering pipeline that cleans and aggregate data into a csv file
        `python process_data.py rawdatafile spark_user_data.csv`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py spark_user_data.csv best_model`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

### File Description

<pre>
.
- app/
  |- data/
  |  |- DisasterResponse.db  #SQLite DB file
  |  |- disaster_categories.csv #Raw category data
  |  |- disaster_messages.csv #Raw message data
  |  |- process_date.py # script to ETL raw data into SQLite DB
  |- trained_model/
  |  |- GBTClassificationModel # model folder
  |  |- LogisticRegressionModel # model folder
  |  |- RandomForestClassificationModel # model folder
  |- templates/
  |  |- model.html # model presentation page
  |  |- master.html # Main page
  |- run.py # Web entry script
  |- process_data.py # pyspark script - transform source data into user level data
  |- train_classifier.py # pyspark ml script - create prediction models
  |- spark_user_data.csv # sample user-level data file
- README.md
- sparkify_data_engineering.ipynb # jupyter notebook - analyzing data, source data, and transform data
- sparkify_modelling.ipynb # jupyter notebook - featuring data and create prediction models.
- capstone-project-writeup.docx # Project Writeup

</pre>

###  Libraries need for Running Code
- pyspark
- pyspark.sql
- pyspark.sql.functions
- pyspark.ml
- pyspark.ml.classification
- pyspark.ml.feature
- pyspark.ml.tuning
- pyspark.ml.evaluation
- sys
- timeit
- datetime
- flask
- plotly.graph_objs
- pandas
- json
- plotly
- findspark

### Note:

- Both jupyter notebook and web application are writen in pyspark. To be able to run the applications and notebook successfully, you will need to have spark installed on your local machine.
- The web application only tested and run in local machine at the moment. In future, it will be hosted on the web. 
- I strongly recommended to run the scripts under conda environment. It's easy to install required python libs.
- The 12GB source file can be found on Udacity S3 Buckets. You can download it to local. However, if it takes too long, a small file is also avaiable on the S3.
- Running Spark ML jobs for creating modelling takes time depends on your machine configurations. However, you can skip the process as I have already save those models under folder 'app/trained_model/'

### Project Github 

https://github.com/zhangshen20/capstone-project-sparkify

