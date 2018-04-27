#
#  Project     : Preceptron
#  File        : random_forest.py
#
#  Description : Library functions for ML algorithm random_forest 
#
#  Input       : Mostly spark_session and dataframe 
#
#  Output      : Trained Model or Predictions
#
#

#from pyspark.sql import SparkSession
#from pyspark.sql.functions import lit
#from pyspark.sql.functions import monotonically_increasing_id
#from pyspark.ml.linalg import Vectors
#from pyspark.sql.functions import regexp_replace, col
#from pyspark.mllib.linalg import SparseVector
#from pyspark.mllib.regression import LabeledPoint
#from pyspark.mllib.linalg import SparseVector
#from pyspark.mllib.util import MLUtils
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Function      : perceptron_prepare_data
# Description   : Convert dataframe coloumns to label and features
# Input         : Spark_session, Dataframe
# Output        : Transformed Dataframe
#
def perceptron_prepare_data(sp_session,df_p):
   assembler = VectorAssembler(
      inputCols=["DST_IP_INDEX","AVG_PKTS", "MAX_PKTS", "AVG_BYTS","MAX_BYTS","AVG_FLOW_CNT"],
      outputCol="features")

   output = assembler.transform(df_p)
   output.select('label',"features").show(truncate=False)
   #output.show()
   return output

# Function      : perceptron_random_forest_train
# Description   : Train with random forest algorithm
# Input         : Spark_session, Dataframe
# Output        : Trained model saved in HDFS
#
def perceptron_random_forest_train(sp_session,df_p):

    data = df_p
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
         VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    #rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers. 
    # Train on complete data
    model = pipeline.fit(data)
    ##model = rf.fit(trainingData)
    
    #Save the model
    model.save("ml_randomforest.model")

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
         labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only

# Function      : perceptron_random_forest_predict
# Description   : Predict with precalculated random forest model
# Input         : Spark_session, Dataframe
# Output        : Predictions saved in HDFS
#
def perceptron_random_forest_predict(sp_session,df_p):
    #model = RandomForestClassificationModel.load("ml_randomforest.model")
    model = PipelineModel.load("ml_randomforest.model")
    predictions = model.transform(df_p)
    predictions.select("predictedLabel", "label", "features").show(5)



