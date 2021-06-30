import sys
from timeit import default_timer as timer
import datetime
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.functions import map_keys

from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Pipeline, pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def load_data(filepath) -> DataFrame:
    """Load dataset from file. Extract datasets and category names
    
    Args:
        filepath (str): user level file path
        
    Return:
        df (DataFrame): DateFrame
    """
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    
    return df

def build_model_pipeline(feature_cols: list) -> CrossValidator:
    """Build ML Pipelines with transformer and estimater. This Function uses 'GBTClassifer' model
       Uses CrossValidator for optimize the models by using a number of parameters - 
        - maxDepth, maxBins, maxIter
    
    Args:
        feature_cols (list): List of columns to be used for featuring
        
    Return:
        CrossValidator: CrossValidator
    """

    stages = []

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_index_assembler")
    stages += [assembler]
    scaler = Normalizer(inputCol=assembler.getOutputCol(), outputCol="features")
    stages += [scaler]
    gbt = GBTClassifier(featuresCol='features', labelCol='label')
    stages += [gbt]

    pipeline = Pipeline(stages = stages)    

    paramGrid = (ParamGridBuilder()
                .addGrid(gbt.maxDepth, [2, 4, 6])
                .addGrid(gbt.maxBins,  [20, 60])
                .addGrid(gbt.maxIter,  [10, 20])
                .build())    

    evaluator = BinaryClassificationEvaluator(predictionCol="prediction")

    gbt_cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

    return gbt_cv

def build_multi_model_pipeline(feature_cols: list): #-> (list, Pipeline):
    """Build 3 ML Pipelines, uses 'GBTClassifer', 'LogisticRegression', 'RandomForestClassifier'
       Uses CrossValidator across the 3 estimators for optimize the models by using a number of parameters - 
        - GBTClassifier -> maxDepth, maxBins, maxIter
        - RandomForestClassifier
        - LogisticRegression -> regParam, elasticNetParam
    
    Args:
        feature_cols (list): List of columns to be used for featuring
        
    Return:
        grid_loop (List of ParamGridBuilder): ParamGridBuilder
        pipeline (Pipeline): Pipeline for training data using all of 3 modellings
    """

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_index_assembler")
    scaler = Normalizer(inputCol=assembler.getOutputCol(), outputCol="features")

    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lr_regParam = [0.01, 0.1, 0.3, 0.5]
    lr_elasticNetParam=[0, .5, 1]

    dt = DecisionTreeClassifier(maxDepth=3)
    dt_maxDepth = [3, 5]

    rf = RandomForestClassifier()

    gbt = GBTClassifier()
    gbt_maxDepth = [2, 4, 6]
    gbt_maxBins = [20, 60]
    gbt_maxIter = [10, 20]

    pipeline = Pipeline(stages = [])
    
    lr_stages = [assembler, scaler, lr]
    lr_paramgrid = ParamGridBuilder().baseOn({pipeline.stages:lr_stages}) \
                                     .addGrid(lr.regParam, lr_regParam) \
                                     .addGrid(lr.elasticNetParam, lr_elasticNetParam) \
                                     .build()

    dt_stages = [assembler, scaler, dt]
    dt_paramgrid = ParamGridBuilder().baseOn({pipeline.stages:dt_stages}) \
                                     .addGrid(dt.maxDepth, dt_maxDepth) \
                                     .build()

    rf_stages = [assembler, scaler, rf]
    rf_paramgrid = ParamGridBuilder().baseOn({pipeline.stages: rf_stages}) \
                                     .build()

    gbt_stages = [assembler, scaler, gbt]                                     
    gbt_paramgrid = ParamGridBuilder().baseOn({pipeline.stages:gbt_stages}) \
                                      .addGrid(gbt.maxDepth, gbt_maxDepth) \
                                      .addGrid(gbt.maxBins, gbt_maxBins) \
                                      .addGrid(gbt.maxIter, gbt_maxIter) \
                                      .build()

    grid_loop = [lr_paramgrid, gbt_paramgrid, rf_paramgrid]
    # grid_loop = [lr_paramgrid, dt_paramgrid, rf_paramgrid, gbt_paramgrid]

    return grid_loop, pipeline

def select_best_model(grid_loop: list, pipeline: Pipeline, df: DataFrame):
    '''
    '''

    # how many parallel threads should the crossvalidator use
    parallelExec=8

    # should the corssvalidator collect submodel data
    trackSubModels=False

    # how many folds should crossvalidator use
    numberFolds=3    

    paramCombo = 0

    for grid in grid_loop:
        paramCombo = paramCombo + len(grid)

    print("Number of parameter combinations being tested ", paramCombo)        
    print("Start Cross Validation at ", datetime.datetime.now(), "\n")

    best_acc = 0
    
    for grid in grid_loop:
        print(">>> ------------------------------------------- <<<")
        # print("Running grid ", grid, "\n")
        starttime = timer()
        cross_val = CrossValidator(estimatorParamMaps=grid, 
                                   estimator=pipeline,  
                                   evaluator=BinaryClassificationEvaluator(),
                                   numFolds=numberFolds,
                                   parallelism=parallelExec,
                                   collectSubModels=trackSubModels)

        cv_model = cross_val.fit(df)
        print("*** Time to cross validation ", timer() - starttime)
        # get the accuracy metrics for the models.
        avg_metrics_grid = cv_model.avgMetrics
        print("*** avg_metrics_grid ***")
        print(avg_metrics_grid)
        # get the max accuracy metic in the list of accuracy metrics
        model_acc = max(avg_metrics_grid)
        print("max score for this grid -> ", model_acc)

        
        model_name = str(cv_model.bestModel.stages[2]).split(":")[0]
        print(cv_model.bestModel.stages)
        print(model_name)

        cv_model.bestModel.save('trained_model/' + model_name)

        if(model_acc > best_acc):
            print("This model has greater accuracy. Old accuracy -> ", best_acc, " | New accuracy -> ", model_acc)
            best_model = cv_model.bestModel
            best_acc = model_acc
            # Print out the parameters for all the stages of this model
            # for stage in best_model.stages:
            #     print(stage.extractParamMap())

    return best_model

if __name__ == '__main__':    

    if len(sys.argv) == 3:
        train_file, model_file = sys.argv[1:]

        conf = SparkConf().setMaster("local[*]").setAppName("multigridsearch")
        sc = SparkContext(conf=conf)
        sc.setLogLevel("ERROR")

        spark = SparkSession(sc)

        # # create a Spark session
        # spark = SparkSession.builder \
        #         .master("local") \
        #         .appName("sparkify_data_modelling") \
        #         .getOrCreate()        

        df = load_data(train_file)

        feature_cols = \
            ['num_active_days',
            'Add Friend',
            'Add to Playlist',
            'Downgrade',
            'Thumbs Up',
            'days_since_registration',
            'avg_songs_per_active_day']        

        df = df.withColumn('label', df.churn)
        splits = df.randomSplit([0.8, 0.2], seed=42)

        df_train = splits[0]
        df_test  = splits[1]        

        grid_loop, my_pipeline = build_multi_model_pipeline(feature_cols)

        best_model = select_best_model(grid_loop, my_pipeline, df_train)
        predictions = best_model.transform(df_test)
        predictions.select('rawPrediction', 'prediction', 'probability').show(10)

        evaluator = BinaryClassificationEvaluator()
        result = evaluator.evaluate(predictions)
        print("Evaluation Result -> ", result)

        best_model.save(model_file)        

    else:
        print('Please provide the source file and model file name. '\
              'Arg 1 is for source file, Arg 2 is for model file')        