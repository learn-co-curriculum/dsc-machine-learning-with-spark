# Machine Learning with Spark

## Introduction

You've now explored how to perform operations on Spark RDDs for simple MapReduce tasks. Luckily, there are far more advanced use cases for Spark, and many of them are found in the `ml` library, which we are going to explore in this lesson.


## Objectives

You will be able to: 

- Load and manipulate data using Spark DataFrames  
- Define estimators and transformers in Spark ML 
- Create a Spark ML pipeline that transforms data and runs over a grid of hyperparameters 



## A Tale of Two Libraries

If you look at the PySpark documentation, you'll notice that there are two different libraries for machine learning, [mllib](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html) and [ml](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html). These libraries are extremely similar to one another, the only difference being that the `mllib` library is built upon the RDDs you just practiced using; whereas, the `ml` library is built on higher level Spark DataFrames, which has methods and attributes similar to pandas. Spark has stated that in the future, it is going to devote more effort to the `ml` library and that `mllib` will become deprecated. It's important to note that these libraries are much younger than pandas and scikit-learn and there are not as many features present in either.

## Spark DataFrames

In the previous lessons, you were introduced to SparkContext as the primary way to connect with a Spark Application. Here, we will be using SparkSession, which is from the [sql](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html) component of PySpark. The SparkSession acts the same way as SparkContext; it is a bridge between Python and the Spark Application. It's just built on top of the Spark SQL API, a higher-level API than RDDs. In fact, a SparkContext object is spun up around which the SparkSession object is wrapped. Let's go through the process of manipulating some data here. For this example, we're going to be using the [Forest Fire dataset](https://archive.ics.uci.edu/ml/datasets/Forest+Fires) from UCI, which contains data about the area burned by wildfires in the Northeast region of Portugal in relation to numerous other factors.

To begin with, let's create a SparkSession so that we can spin up our spark application. 


```python
# importing the necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
# sc = SparkContext('local[*]')
# spark = SparkSession(sc)
```

To create a SparkSession: 


```python
spark = SparkSession.builder.master('local').getOrCreate()
```

Now, we'll load the data into a PySpark DataFrame: 


```python
## reading in pyspark df
spark_df = spark.read.csv('./forestfires.csv', header='true', inferSchema='true')

## observing the datatype of df
type(spark_df)
```




    pyspark.sql.dataframe.DataFrame



You'll notice that some of the methods are extremely similar or the same as those found within Pandas.


```python
spark_df.head()
```




    Row(X=7, Y=5, month='mar', day='fri', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0)




```python
spark_df.columns
```




    ['X',
     'Y',
     'month',
     'day',
     'FFMC',
     'DMC',
     'DC',
     'ISI',
     'temp',
     'RH',
     'wind',
     'rain',
     'area']



Selecting multiple columns is similar as well: 


```python
spark_df[['month','day','rain']]
```




    DataFrame[month: string, day: string, rain: double]



But selecting one column is different. If you want to maintain the methods of a spark DataFrame, you should use the `.select()` method. If you want to just select the column, you can use the same method you would use in pandas (this is primarily what you would use if you're attempting to create a boolean mask). 


```python
d = spark_df.select('rain')
```


```python
spark_df['rain']
```




    Column<'rain'>



Let's take a look at all of our data types in this dataframe


```python
spark_df.dtypes
```




    [('X', 'int'),
     ('Y', 'int'),
     ('month', 'string'),
     ('day', 'string'),
     ('FFMC', 'double'),
     ('DMC', 'double'),
     ('DC', 'double'),
     ('ISI', 'double'),
     ('temp', 'double'),
     ('RH', 'int'),
     ('wind', 'double'),
     ('rain', 'double'),
     ('area', 'double')]



## Aggregations with our DataFrame

Let's investigate to see if there is any correlation between what month it is and the area of fire: 


```python
spark_df_months = spark_df.groupBy('month').agg({'area': 'mean'})
spark_df_months
```




    DataFrame[month: string, avg(area): double]



Notice how the grouped DataFrame is not returned when you call the aggregation method. Remember, this is still Spark! The transformations and actions are kept separate so that it is easier to manage large quantities of data. You can perform the transformation by calling `.collect()`: 


```python
spark_df_months.collect()
```




    [Row(month='jun', avg(area)=5.841176470588234),
     Row(month='aug', avg(area)=12.489076086956521),
     Row(month='may', avg(area)=19.24),
     Row(month='feb', avg(area)=6.275),
     Row(month='sep', avg(area)=17.942616279069753),
     Row(month='mar', avg(area)=4.356666666666667),
     Row(month='oct', avg(area)=6.638),
     Row(month='jul', avg(area)=14.3696875),
     Row(month='nov', avg(area)=0.0),
     Row(month='apr', avg(area)=8.891111111111112),
     Row(month='dec', avg(area)=13.33),
     Row(month='jan', avg(area)=0.0)]



As you can see, there seem to be larger area fires during what would be considered the summer months in Portugal. On your own, practice more aggregations and manipulations that you might be able to perform on this dataset. 

## Boolean Masking 

Boolean masking also works with PySpark DataFrames just like Pandas DataFrames, the only difference being that the `.filter()` method is used in PySpark. To try this out, let's compare the amount of fire in those areas with absolutely no rain to those areas that had rain.


```python
no_rain = spark_df.filter(spark_df['rain'] == 0.0)
some_rain = spark_df.filter(spark_df['rain'] > 0.0)
```

Now, to perform calculations to find the mean of a column, we'll have to import functions from `pyspark.sql`. As always, to read more about them, check out the [documentation](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions).


```python
from pyspark.sql.functions import mean

print('no rain fire area: ', no_rain.select(mean('area')).show(),'\n')

print('some rain fire area: ', some_rain.select(mean('area')).show(),'\n')
```

    +------------------+
    |         avg(area)|
    +------------------+
    |13.023693516699408|
    +------------------+
    
    no rain fire area:  None 
    
    +---------+
    |avg(area)|
    +---------+
    |  1.62375|
    +---------+
    
    some rain fire area:  None 
    


Yes there's definitely something there! Unsurprisingly, rain plays in a big factor in the spread of wildfire.

Let's obtain data from only the summer months in Portugal (June, July, and August). We can also do the same for the winter months in Portugal (December, January, February).


```python
summer_months = spark_df.filter(spark_df['month'].isin(['jun','jul','aug']))
winter_months = spark_df.filter(spark_df['month'].isin(['dec','jan','feb']))

print('summer months fire area', summer_months.select(mean('area')).show())
print('winter months fire areas', winter_months.select(mean('area')).show())
```

    +------------------+
    |         avg(area)|
    +------------------+
    |12.262317596566525|
    +------------------+
    
    summer months fire area None
    +-----------------+
    |        avg(area)|
    +-----------------+
    |7.918387096774193|
    +-----------------+
    
    winter months fire areas None


## Machine Learning

Now that we've performed some data manipulation and aggregation, lets get to the really cool stuff, machine learning! PySpark states that they've used scikit-learn as an inspiration for their implementation of a machine learning library. As a result, many of the methods and functionalities look similar, but there are some crucial distinctions. There are three main concepts found within the ML library:

`Transformer`: An algorithm that transforms one PySpark DataFrame into another DataFrame. 

`Estimator`: An algorithm that can be fit onto a PySpark DataFrame that can then be used as a Transformer. 

`Pipeline`: A pipeline very similar to an `sklearn` pipeline that chains together different actions.

The reasoning behind this separation of the fitting and transforming step is because Spark is lazily evaluated, so the 'fitting' of a model does not actually take place until the Transformation action is called. Let's examine what this actually looks like by performing a regression on the Forest Fire dataset. To start off with, we'll import the necessary libraries for our tasks.


```python
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import feature
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
```

Looking at our data, one can see that all the categories are numerical except for day and month. We saw some correlation between the month and area burned in a fire, so we will include that in our model. The day of the week, however, is highly unlikely to have any effect on fire, so we will drop it from the DataFrame.


```python
fire_df = spark_df.drop('day')
fire_df.head()
```




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0)



In order for us to run our model, we need to turn the months variable into a dummy variable. In `ml` this is a 2-step process that first requires turning the categorical variable into a numerical index (`StringIndexer`). Only after the variable is an integer can PySpark create dummy variable columns related to each category (`OneHotEncoder`). Your key parameters when using these `ml` estimators are: `inputCol` (the column you want to change) and `outputCol` (where you will store the changed column). Here it is in action: 


```python
si = StringIndexer(inputCol='month', outputCol='month_num')
model = si.fit(fire_df)
new_df = model.transform(fire_df)
```

Note the small, but critical distinction between `sklearn`'s implementation of a transformer and PySpark's implementation. `sklearn` is more object oriented and Spark is more functional oriented.


```python
## this is an estimator (an untrained transformer)
type(si)
```




    pyspark.ml.feature.StringIndexer




```python
## this is a transformer (a trained transformer)
type(model)
```




    pyspark.ml.feature.StringIndexerModel




```python
model.labels
```




    ['aug',
     'sep',
     'mar',
     'jul',
     'feb',
     'jun',
     'oct',
     'apr',
     'dec',
     'jan',
     'may',
     'nov']




```python
new_df.head(4)
```




    [Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0),
     Row(X=7, Y=4, month='oct', FFMC=90.6, DMC=35.4, DC=669.1, ISI=6.7, temp=18.0, RH=33, wind=0.9, rain=0.0, area=0.0, month_num=6.0),
     Row(X=7, Y=4, month='oct', FFMC=90.6, DMC=43.7, DC=686.9, ISI=6.7, temp=14.6, RH=33, wind=1.3, rain=0.0, area=0.0, month_num=6.0),
     Row(X=8, Y=6, month='mar', FFMC=91.7, DMC=33.3, DC=77.5, ISI=9.0, temp=8.3, RH=97, wind=4.0, rain=0.2, area=0.0, month_num=2.0)]



As you can see, we have created a new column called `'month_num'` that represents the month by a number. Now that we have performed this step, we can use Spark's version of `OneHotEncoder()` Let's make sure we have an accurate representation of the months.


```python
new_df.select('month_num').distinct().collect()
```




    [Row(month_num=8.0),
     Row(month_num=0.0),
     Row(month_num=7.0),
     Row(month_num=1.0),
     Row(month_num=4.0),
     Row(month_num=11.0),
     Row(month_num=3.0),
     Row(month_num=2.0),
     Row(month_num=10.0),
     Row(month_num=6.0),
     Row(month_num=5.0),
     Row(month_num=9.0)]




```python
## fitting and transforming the OneHotEncoder
ohe = feature.OneHotEncoder(inputCols=['month_num'], outputCols=['month_vec'], dropLast=True)
one_hot_encoded = ohe.fit(new_df).transform(new_df)
one_hot_encoded.head()
```




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0, month_vec=SparseVector(11, {2: 1.0}))



Great, we now have a OneHotEncoded sparse vector in the `'month_vec'` column! Because Spark is optimized for big data, sparse vectors are used rather than entirely new columns for dummy variables because it is more space efficient. You can see in this first row of the DataFrame:  

`month_vec=SparseVector(11, {2: 1.0})` this indicates that we have a sparse vector of size 11 (because of the parameter `dropLast = True` in `OneHotEncoder()`) and this particular data point is the 2nd index of our month labels (march, based off the labels in the `model` StringEstimator transformer).  

The final requirement for all machine learning models in PySpark is to put all of the features of your model into one sparse vector. This is once again for efficiency sake. Here, we are doing that with the `VectorAssembler()` estimator.


```python
features = ['X',
 'Y',
 'FFMC',
 'DMC',
 'DC',
 'ISI',
 'temp',
 'RH',
 'wind',
 'rain',
 'month_vec']

target = 'area'

vector = VectorAssembler(inputCols=features, outputCol='features')
vectorized_df = vector.transform(one_hot_encoded)
```


```python
vectorized_df.head()
```




    Row(X=7, Y=5, month='mar', FFMC=86.2, DMC=26.2, DC=94.3, ISI=5.1, temp=8.2, RH=51, wind=6.7, rain=0.0, area=0.0, month_num=2.0, month_vec=SparseVector(11, {2: 1.0}), features=SparseVector(21, {0: 7.0, 1: 5.0, 2: 86.2, 3: 26.2, 4: 94.3, 5: 5.1, 6: 8.2, 7: 51.0, 8: 6.7, 12: 1.0}))



Great! We now have our data in a format that seems acceptable for the last step. It's time for us to actually fit our model to data! Let's fit a Random Forest Regression model to our data. Although there are still a bunch of other features in the DataFrame, it doesn't matter for the machine learning model API. All that needs to be specified are the names of the features column and the label column. 


```python
## instantiating and fitting the model
rf_model = RandomForestRegressor(featuresCol='features', 
                                 labelCol='area', predictionCol='prediction').fit(vectorized_df)
```


```python
rf_model.featureImportances
```




    SparseVector(21, {0: 0.1239, 1: 0.0592, 2: 0.0336, 3: 0.1819, 4: 0.0707, 5: 0.1562, 6: 0.1224, 7: 0.1576, 8: 0.0454, 9: 0.0, 10: 0.0045, 11: 0.0387, 12: 0.001, 13: 0.0039, 14: 0.0003, 15: 0.0002, 16: 0.0, 17: 0.0001, 18: 0.0001, 20: 0.0005})




```python
## generating predictions
predictions = rf_model.transform(vectorized_df).select('area', 'prediction')
predictions.head(10)
```




    [Row(area=0.0, prediction=6.169838669102178),
     Row(area=0.0, prediction=6.1500932021665315),
     Row(area=0.0, prediction=6.628494837350649),
     Row(area=0.0, prediction=5.836624400829339),
     Row(area=0.0, prediction=6.097209996231621),
     Row(area=0.0, prediction=8.031313609213454),
     Row(area=0.0, prediction=3.8486266574277677),
     Row(area=0.0, prediction=10.357114627905283),
     Row(area=0.0, prediction=11.48631588473679),
     Row(area=0.0, prediction=4.656677887252801)]



Now we can evaluate how well the model performed using `RegressionEvaluator`.


```python
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area')
```


```python
## evaluating r^2
evaluator.evaluate(predictions,{evaluator.metricName: 'r2'})
```




    0.7672059152947321




```python
## evaluating mean absolute error
evaluator.evaluate(predictions,{evaluator.metricName: 'mae'})
```




    13.571402440225649



## Putting it all in a Pipeline

We just performed a whole lot of transformations to our data. Let's take a look at all the estimators we used to create this model:

* `StringIndexer()` 
* `OneHotEnconder()` 
* `VectorAssembler()` 
* `RandomForestRegressor()` 

Once we've fit our model in the Pipeline, we're then going to want to evaluate it to determine how well it performs. We can do this with:

* `RegressionEvaluator()` 

We can streamline all of these transformations to make it much more efficient by chaining them together in a pipeline. The Pipeline object expects a list of the estimators prior set to the parameter `stages`.


```python
# importing relevant libraries
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml import Pipeline
```


```python
## instantiating all necessary estimator objects

string_indexer = StringIndexer(inputCol='month', outputCol='month_num', handleInvalid='keep')
one_hot_encoder = OneHotEncoder(inputCols=['month_num'], outputCols=['month_vec'], dropLast=True)
vector_assember = VectorAssembler(inputCols=features, outputCol='features')
random_forest = RandomForestRegressor(featuresCol='features', labelCol='area')
stages = [string_indexer, one_hot_encoder, vector_assember, random_forest]

# instantiating the pipeline with all them estimator objects
pipeline = Pipeline(stages=stages)
```

### Cross-validation 

You might have missed a critical step in the random forest regression above; we did not cross validate or perform a train/test split! Now we're going to fix that by performing cross-validation and also testing out multiple different combinations of parameters in PySpark's `GridSearch()` equivalent. To begin with, we will create a parameter grid that contains the different parameters we want to use in our model.


```python
# creating parameter grid

params = ParamGridBuilder()\
          .addGrid(random_forest.maxDepth, [5, 10, 15])\
          .addGrid(random_forest.numTrees, [20 ,50, 100])\
          .build()
```

Let's take a look at the params variable we just built.


```python
print('total combinations of parameters: ', len(params))

params[0]
```

    total combinations of parameters:  9





    {Param(parent='RandomForestRegressor_f6cb8e7e34a2', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5,
     Param(parent='RandomForestRegressor_f6cb8e7e34a2', name='numTrees', doc='Number of trees to train (>= 1).'): 20}



Now it's time to combine all the steps we've created to work in a single line of code with the `CrossValidator()` estimator.


```python
## instantiating the evaluator by which we will measure our model's performance
reg_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='area', metricName = 'mae')

## instantiating crossvalidator estimator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=reg_evaluator, parallelism=4)
```


```python
## fitting crossvalidator
cross_validated_model = cv.fit(fire_df)
```

Now, let's see how well the model performed! Let's take a look at the average performance for each one of our 9 models. It looks like the optimal performance is an MAE around 23. Note that this is worse than our original model, but that's because our original model had substantial data leakage. We didn't do a train-test split!


```python
cross_validated_model.avgMetrics
```




    [20.31862782870068,
     21.035939448928634,
     20.531510266985368,
     20.923948890077472,
     21.531377089474567,
     21.231370160439937,
     20.995655577183282,
     21.700103553698707,
     21.28433354711881]



Now, let's take a look at the optimal parameters of our best performing model. The `cross_validated_model` variable is now saved as the best performing model from the grid search just performed. Let's look to see how well the predictions performed. As you can see, this dataset has a large number of areas of "0.0" burned. Perhaps, it would be better to investigate this problem as a classification task.


```python
predictions = cross_validated_model.transform(spark_df)
predictions.select('prediction', 'area').show(300)
```

    +------------------+-------+
    |        prediction|   area|
    +------------------+-------+
    | 7.534336534011331|    0.0|
    | 5.472512273177187|    0.0|
    | 5.766199773177187|    0.0|
    |  9.63133836943096|    0.0|
    | 4.267483671509217|    0.0|
    | 6.880669578310477|    0.0|
    |   6.0136521555355|    0.0|
    |14.655464669032536|    0.0|
    |10.081704879169665|    0.0|
    | 6.141124179277453|    0.0|
    | 5.653560914487523|    0.0|
    | 4.992702444003092|    0.0|
    | 5.822982906740969|    0.0|
    |10.060852422127201|    0.0|
    |180.06464366625673|    0.0|
    | 8.040165959538426|    0.0|
    | 6.295140722295841|    0.0|
    | 8.230844864377461|    0.0|
    |4.7138664207345675|    0.0|
    | 5.463950973807268|    0.0|
    |15.411168889821244|    0.0|
    |3.2276869403290576|    0.0|
    | 5.322530774040546|    0.0|
    |24.118000479159555|    0.0|
    | 8.738970540796545|    0.0|
    | 7.351729176788401|    0.0|
    | 8.417971024167176|    0.0|
    |11.811350088787524|    0.0|
    |62.121768262357804|    0.0|
    | 8.638860368509011|    0.0|
    | 59.93967757604023|    0.0|
    | 6.475771020476371|    0.0|
    | 4.378253771599083|    0.0|
    |4.8860163228364195|    0.0|
    | 4.771008108917472|    0.0|
    | 6.757248446571562|    0.0|
    | 6.028033704947967|    0.0|
    | 6.622013250604056|    0.0|
    | 7.372836633090851|    0.0|
    | 4.691222307258402|    0.0|
    |16.846051168229298|    0.0|
    | 4.505990570306316|    0.0|
    | 3.515328540626671|    0.0|
    | 4.597749652918525|    0.0|
    | 6.167538984576884|    0.0|
    | 223.2191618547517|    0.0|
    | 7.115855828852214|    0.0|
    | 3.275846919341286|    0.0|
    |4.0649686365386435|    0.0|
    | 5.549120909278674|    0.0|
    | 6.999158991776244|    0.0|
    |  4.04696173337114|    0.0|
    |  4.64990268640381|    0.0|
    |  4.64990268640381|    0.0|
    | 4.125828089555562|    0.0|
    | 6.448455048610759|    0.0|
    | 5.964548041532999|    0.0|
    |  5.46900595596872|    0.0|
    | 5.450008992056958|    0.0|
    |10.197246908989722|    0.0|
    | 4.357554062040851|    0.0|
    | 9.215828874287826|    0.0|
    |4.7960653300727225|    0.0|
    |3.6506787586890694|    0.0|
    | 3.555546191874682|    0.0|
    |  9.77779837908617|    0.0|
    | 15.11335364720028|    0.0|
    | 17.34749083237353|    0.0|
    | 17.34749083237353|    0.0|
    | 4.349690437009271|    0.0|
    | 5.475663800886926|    0.0|
    | 4.093642959705793|    0.0|
    | 4.913865112051743|    0.0|
    | 5.894712369242929|    0.0|
    |  8.80720062212929|    0.0|
    | 6.381825147287571|    0.0|
    | 6.498014404455333|    0.0|
    | 5.736430931694762|    0.0|
    | 3.440588953197468|    0.0|
    |26.718891585375882|    0.0|
    | 7.780264715544351|    0.0|
    |14.938163248345585|    0.0|
    | 6.150266020496304|    0.0|
    |13.219105988720292|    0.0|
    | 7.818690632774766|    0.0|
    | 5.066724950901609|    0.0|
    | 2.616068143848855|    0.0|
    |4.9347110288908365|    0.0|
    |11.755751609060113|    0.0|
    | 5.992902764606158|    0.0|
    | 5.463538944209156|    0.0|
    | 8.933123677333274|    0.0|
    | 6.277335550565971|    0.0|
    |25.233881625389646|    0.0|
    | 8.154255703599055|    0.0|
    |5.3977414475796275|    0.0|
    | 5.341804606672054|    0.0|
    | 5.872550472542931|    0.0|
    | 4.972767544206854|    0.0|
    | 6.204232240649717|    0.0|
    | 6.204232240649717|    0.0|
    | 5.255063960702317|    0.0|
    | 5.177824123955582|    0.0|
    | 8.347875225617875|    0.0|
    | 5.026861511496833|    0.0|
    | 5.549120909278674|    0.0|
    |4.5081358387641135|    0.0|
    |  4.04696173337114|    0.0|
    | 4.761170705550105|    0.0|
    |5.7433327637552205|    0.0|
    | 5.549120909278674|    0.0|
    |3.5776966241889014|    0.0|
    | 4.762606552185608|    0.0|
    |3.3079108274038527|    0.0|
    | 5.550381508024396|    0.0|
    | 5.550381508024396|    0.0|
    | 4.739491880052629|    0.0|
    | 5.530503273944335|    0.0|
    |  4.32144950845959|    0.0|
    | 3.601552764965782|    0.0|
    | 5.183438283669794|    0.0|
    | 5.701376841293947|    0.0|
    | 8.153287064422468|    0.0|
    |12.890855289852464|    0.0|
    | 5.255423945321739|    0.0|
    | 5.134095109403597|    0.0|
    | 4.528851245266532|    0.0|
    | 7.862061705713657|    0.0|
    | 4.178804366682578|    0.0|
    | 3.987726507123937|    0.0|
    | 4.909422310407624|    0.0|
    |4.0649686365386435|    0.0|
    |  4.64280522066793|    0.0|
    | 4.377800796136571|    0.0|
    | 3.902597574546774|    0.0|
    | 6.758102284367787|    0.0|
    | 11.14856713416727|    0.0|
    | 9.179749902926288|    0.0|
    | 6.679827308524665|   0.36|
    | 12.25141596913038|   0.43|
    |  9.01551529289925|   0.47|
    | 4.365483000758387|   0.55|
    | 9.119018157029007|   0.61|
    |13.334659426076499|   0.71|
    |3.3647164832653247|   0.77|
    |10.144040414787103|    0.9|
    |18.413041151602396|   0.95|
    |  13.1858449746935|   0.96|
    | 4.444742156441039|   1.07|
    | 8.889762512346739|   1.12|
    | 5.602818704388446|   1.19|
    |28.346949860849215|   1.36|
    |13.047906561595786|   1.43|
    | 5.448523945321738|   1.46|
    |50.095557594757416|   1.46|
    |3.9021048805503598|   1.56|
    |53.280284415485724|   1.61|
    |6.0102134429298895|   1.63|
    | 4.118431592573716|   1.64|
    | 8.417971024167176|   1.69|
    | 4.760436397798514|   1.75|
    |5.6158062283716115|    1.9|
    | 6.911855625483099|   1.94|
    |18.720665637960437|   1.95|
    | 6.543206057834814|   2.01|
    | 6.296189368154659|   2.14|
    |3.7116899415071587|   2.29|
    | 7.461138335747253|   2.51|
    | 10.59987698589714|   2.53|
    |  8.24321462850322|   2.55|
    | 6.429111531525931|   2.57|
    | 7.097574065445238|   2.69|
    | 9.795125322662873|   2.74|
    | 9.468678678648978|   3.07|
    | 5.283041315347509|    3.5|
    | 5.976376341360561|   4.53|
    | 6.401558644002047|   4.61|
    | 5.104327201554016|   4.69|
    | 6.179908185943126|   4.88|
    | 15.82919914582194|   5.23|
    | 7.685691499930121|   5.33|
    |13.885261785727248|   5.44|
    |  6.69971127472453|   6.38|
    | 7.697702381483062|   6.83|
    | 8.751160931669101|   6.96|
    |10.632066772289585|   7.04|
    |16.926466119058546|   7.19|
    |11.175146653632016|    7.3|
    | 4.585330091054062|    7.4|
    |  6.77486185857079|   8.24|
    |10.180397123526934|   8.31|
    | 6.983581210839006|   8.68|
    | 7.379217841280727|   8.71|
    | 15.79593394671347|   9.41|
    | 7.379217841280727|  10.01|
    | 4.340467225272748|  10.02|
    | 6.401558644002047|  10.93|
    |10.784825658442909|  11.06|
    | 7.796204387903138|  11.24|
    |12.529699866144274|  11.32|
    |13.428897649371809|  11.53|
    | 5.572734179581622|   12.1|
    | 6.228196533804057|  13.05|
    |  8.95192506550853|   13.7|
    | 5.547056879771149|  13.99|
    | 7.009854444560548|  14.57|
    | 6.255581681939147|  15.45|
    |12.696682238739772|   17.2|
    | 8.782800885278288|  19.23|
    | 9.280976619399816|  23.41|
    | 6.014478331402227|  24.23|
    | 9.006315292851534|   26.0|
    | 6.867425275573444|  26.13|
    |    8.116031325678|  27.35|
    | 6.295140722295841|  28.66|
    | 6.295140722295841|  28.66|
    | 11.14856713416727|  29.48|
    | 7.468875479253913|  30.32|
    | 16.89576735279507|  31.72|
    | 5.675798909240731|  31.86|
    | 6.180053049304041|  32.07|
    | 9.118325253647749|  35.88|
    | 6.130815700945342|  36.85|
    |21.489693098835467|  37.02|
    | 8.774872880618066|  37.71|
    | 9.049124771691158|  48.55|
    |11.265035714816396|  49.37|
    |15.051163350929357|   58.3|
    |  73.5692931371562|   64.1|
    |16.361379458897044|   71.3|
    |40.396874058413594|  88.49|
    | 70.06421267497133|  95.18|
    |16.180581999068202| 103.39|
    |54.144495486457984| 105.66|
    | 129.8465685709096| 154.88|
    |49.635377965698005| 196.48|
    | 83.16650369021316| 200.94|
    | 64.24381457939371| 212.88|
    | 774.4844145346042|1090.84|
    | 5.257840712730643|    0.0|
    |5.7947747726343835|    0.0|
    | 5.820820659067869|    0.0|
    | 5.524560448806901|  10.13|
    | 8.505812169562764|    0.0|
    | 6.766724085334718|   2.87|
    |14.240249473260107|   0.76|
    |  13.3242125684982|   0.09|
    | 4.300184077011682|   0.75|
    | 5.507613638265471|    0.0|
    | 5.751056023128434|   2.47|
    | 9.954723200662997|   0.68|
    | 5.562121754346206|   0.24|
    | 4.832018547441201|   0.21|
    |5.3104885390053775|   1.52|
    | 9.349186275306874|  10.34|
    | 8.109991437198087|    0.0|
    | 9.011889827523563|   8.02|
    |  5.18627709295074|   0.68|
    | 5.004394093241559|    0.0|
    | 6.273697956783943|   1.38|
    | 4.353003050744756|   8.85|
    |3.1871393252637104|    3.3|
    | 4.154329811711453|   4.25|
    | 9.510949877050113|   1.56|
    | 5.689018783557017|   6.54|
    | 6.747222251737535|   0.79|
    | 8.055011565322722|   0.17|
    | 10.54548008423322|    0.0|
    | 4.892084676868263|    0.0|
    | 7.171821086109071|    4.4|
    | 17.49729178978848|   0.52|
    |   9.9785491053069|   9.27|
    | 6.647491546568136|   3.09|
    | 5.575883357912482|   8.98|
    | 8.008210318452404|  11.19|
    |6.3813593036260645|   5.38|
    | 9.473094306072397|  17.85|
    | 8.490493992843957|  10.73|
    | 9.473094306072397|  22.03|
    | 9.473094306072397|   9.77|
    |6.3813593036260645|   9.27|
    | 9.249691741969832|  24.77|
    | 7.556998638235845|    0.0|
    |   5.9110865420129|    1.1|
    | 7.768684216920754|  24.24|
    | 8.419415533533346|    0.0|
    |10.672860399851633|    0.0|
    | 5.835577082064235|    0.0|
    | 5.641788683567251|    0.0|
    | 5.737372966661388|    0.0|
    | 6.053046494630204|    0.0|
    | 10.12353762081467|    8.0|
    |3.9493449959451317|   2.64|
    | 34.91977214708024|  86.45|
    | 5.020656316912374|   6.57|
    |10.906727507951526|    0.0|
    | 4.962681976059534|    0.9|
    |3.6142361264441227|    0.0|
    | 28.32463474516826|    0.0|
    | 5.001820370406924|    0.0|
    +------------------+-------+
    only showing top 300 rows
    


Now let's go ahead and take a look at the feature importances of our random forest model. In order to do this, we need to unroll our pipeline to access the random forest model. Let's start by first checking out the `.bestModel` attribute of our `cross_validated_model`. 


```python
type(cross_validated_model.bestModel)
```




    pyspark.ml.pipeline.PipelineModel



`ml` is treating the entire pipeline as the best performing model, so we need to go deeper into the pipeline to access the random forest model within it. Previously, we put the random forest model as the final "stage" in the stages variable list. Let's look at the `.stages` attribute of the `.bestModel`.


```python
cross_validated_model.bestModel.stages
```




    [StringIndexerModel: uid=StringIndexer_fc482d544f0d, handleInvalid=keep,
     OneHotEncoderModel: uid=OneHotEncoder_24b81a98aada, dropLast=true, handleInvalid=error, numInputCols=1, numOutputCols=1,
     VectorAssembler_4c6763d92206,
     RandomForestRegressionModel: uid=RandomForestRegressor_f6cb8e7e34a2, numTrees=20, numFeatures=22]



Perfect! There's the RandomForestRegressionModel, represented by the last item in the stages list. Now, we should be able to access all the attributes of the random forest regressor.


```python
optimal_rf_model = cross_validated_model.bestModel.stages[3]
```


```python
optimal_rf_model.featureImportances
```




    SparseVector(22, {0: 0.1684, 1: 0.0724, 2: 0.0632, 3: 0.1486, 4: 0.0628, 5: 0.133, 6: 0.118, 7: 0.084, 8: 0.1194, 10: 0.0019, 11: 0.023, 13: 0.0045, 14: 0.0, 15: 0.0003, 16: 0.0, 17: 0.0001, 18: 0.0, 20: 0.0003})




```python
optimal_rf_model.getNumTrees
```




    20



## Summary

In this lesson, you learned about PySpark's DataFrames, machine learning models, and pipelines. With the use of a pipeline, you can train a huge number of models simultaneously, saving you a substantial amount of time and effort. Up next, you will have a chance to build a PySpark machine learning pipeline of your own with a classification problem!
