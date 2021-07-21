import sys
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Table

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession, column
from pyspark.sql.functions import col

from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml import PipelineModel
import tempfile


# sys.path.insert(1, '../models')
app = Flask(__name__)

training_file = 'spark_user_data.csv'

# conf = SparkConf().setMaster("local[*]").setAppName("webApp") #.set("spark.sql.shuffle.partitions", 4)
# sc = SparkContext(conf=conf)
# sc.setLogLevel("ERROR")

# spark = SparkSession(sc)

spark = SparkSession \
        .builder \
        .appName("WebApp") \
        .config("spark.sql.shuffle.partitions", 2) \
        .getOrCreate()

df = spark.read.csv(training_file, header=True, inferSchema=True)
df = df.withColumn('label', df.churn)

gbt_model = PipelineModel.load('trained_model/GBTClassificationModel')
lr_model = PipelineModel.load('trained_model/LogisticRegressionModel')
rf_model = PipelineModel.load('trained_model/RandomForestClassificationModel')

df_pd = pd.read_csv(training_file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)


    # df.head()

    page_sum_label=[ 'About', 
                'Add Friend', 
                'Add to Playlist', 
                'Cancel', 
                'Cancellation Confirmation', 
                'Downgrade', 
                'Error', 
                'Help', 
                'Home', 
                'Login', 
                'Logout', 
                'NextSong', 
                'Register',	
                'Roll Advert', 
                'Save Settings', 
                'Settings', 
                'Submit Downgrade', 
                'Submit Registration', 
                'Submit Upgrade', 
                'Thumbs Down', 
                'Thumbs Up', 
                'Upgrade']

    page_sum_value=df_pd.sum()[page_sum_label]

    page_sum_label_excl_next_song=list(set(page_sum_label)-set(['NextSong']))
    page_sum_value_excl_next_song=df_pd.sum()[page_sum_label_excl_next_song]

    churn_counts = df_pd.groupby('churn').count()['userId']
    churn_name = list(['Not Churned', 'Churned'])

    x_label = ['num_active_days_paid', 'num_active_days', 'days_since_registration', ]
    churned = df_pd[df_pd['churn'] == 1].mean()[x_label]
    not_churned = df_pd[df_pd['churn'] == 0].mean()[x_label]    

    # Create a dataframe for the users who have churned out
    df_churn = df_pd[df_pd['churn'] == 1]
    # Create a dataframe for the users who are still in member
    df_not_churn = df_pd[df_pd['churn'] == 0]

    df_between = (df_not_churn.describe() - df_churn.describe()).transpose()


    print("--------------------------")
    print(df_between.columns)
    print(df_between[df_between.columns])
    print("--------------------------")

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(                    
                    x=page_sum_label,
                    y=page_sum_value
                )
            ],
            'layout': {
                'title': 'Page Count',
                'yaxis': {
                    'title': "Page Count"
                },
                'xaxis': {
                    'title': "Page"
                }
            }
        },
        {
            'data': [
                Bar(                    
                    x=page_sum_label_excl_next_song,
                    y=page_sum_value_excl_next_song
                )
            ],
            'layout': {
                'title': 'Page Count Excluding "NextSong"',
                'yaxis': {
                    'title': "Page Count"
                },
                'xaxis': {
                    'title': "Page"
                }
            }
        },
        {
            'data': [
                Pie(
                    values=churn_counts,
                    labels=churn_name
                )
            ]
            ,
            'layout': {
                'title': 'Number of Users - Churned vs Not Churned'
            }
        },
        {
            'data': [
                Table(                    
                    header=dict(values=['', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                    ,font=dict(size=11),align="left"
                    )
                    ,
                    cells=dict(values=[df_between.index, df_between['count'], df_between['mean'], df_between['std'], df_between['min'], df_between['25%'], df_between['50%'], df_between['75%'], df_between['max']]
                    ,font=dict(size=9),align="left"
                    )
                )
            ]
            ,
            'layout': {
                'height': 1200,
                'showlegend': False,
                'title': "Difference Metrics values After Not Churned Minus Churned"
            }
        },        
        {
            'data': [
                Bar(
                    name='Not Churned',
                    x=x_label,
                    y=not_churned
                ),
                Bar(
                    name='Churned',
                    x=x_label,
                    y=churned
                )
            ],
            'layout': {
                'title': 'Registration Days - Churned vs Not Churned',
                'yaxis': {
                    'title': "Registration Days"
                },
                'xaxis': {
                    'title': "Churn"
                },
                'barmode': 'group'
            }
        }        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/model')
def model():
    # save user input in query
    # cvModelRead = CrossValidatorModel.read().load('trained_model/GBTClassificationModel')
    # gbt_model = PipelineModel.load('trained_model/GBTClassificationModel')
    # lr_model = PipelineModel.load('trained_model/LogisticRegressionModel')
    # rf_model = PipelineModel.load('trained_model/RandomForestClassificationModel')

    splits = df.randomSplit([0.9, 0.1], seed=42)

    df_train = splits[0]
    df_test  = splits[1]        

    gbt_predictions = gbt_model.transform(df_test)
    lr_predictions = lr_model.transform(df_test)
    rf_predictions = rf_model.transform(df_test)

    gbt_df_out = gbt_predictions.select('userId', col('prediction').alias('gbt_pred'), 'churn')
    lr_df_out = lr_predictions.select('userId', col('prediction').alias('lr_pred'))
    rf_df_out = rf_predictions.select('userId', col('prediction').alias('rf_pred'))

    df_out = gbt_df_out.join(lr_df_out, on='userId', how='inner') \
                       .join(rf_df_out, on='userId', how='inner') \
                       .select(gbt_df_out.userId, gbt_df_out.gbt_pred, lr_df_out.lr_pred, rf_df_out.rf_pred, gbt_df_out.churn).toPandas()

    gbt_evaluator = BinaryClassificationEvaluator()
    gbt_result = gbt_evaluator.evaluate(gbt_predictions)

    lr_evaluator = BinaryClassificationEvaluator()
    lr_result = lr_evaluator.evaluate(lr_predictions)

    rf_evaluator = BinaryClassificationEvaluator()
    rf_result = rf_evaluator.evaluate(rf_predictions)

    # print("Evaluation Result -> ", rf_result)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(                    
                    x=['gbt', 'lr', 'rf'],
                    y=[gbt_result, lr_result, rf_result]
                )
            ],
            'layout': {
                'title': 'Evaluation Results of 3 Models',
                'yaxis': {
                    'title': "Evaluation Result"
                },
                'xaxis': {
                    'title': "Model"
                }
            }
        },        
        {
            'data': [
                Table(                    
                    header=dict(values=['userId', 'gbt_pred', 'lr_pred', 'rf_pred', 'churn']
                    ,font=dict(size=11),align="left"
                    )
                    ,
                    cells=dict(values=[df_out[x] for x in df_out.columns]
                    ,font=dict(size=9),align="left"
                    )
                )
            ]
            ,
            'layout': {
                'height': 1200,
                'showlegend': False,
                'title': "Test data Predictions on 3 Models"
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('model.html', ids=ids, graphJSON=graphJSON)

def main():
    app.run(host='localhost', port=3001, debug=True)

if __name__ == '__main__':

    main()