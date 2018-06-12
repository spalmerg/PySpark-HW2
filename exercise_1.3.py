from pyspark import SparkContext
from pyspark.sql import HiveContext
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
import datetime



if __name__ == "__main__":
      # set up drivers
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = HiveContext(sc)

    # read in data
    file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
        .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/venmo/venmoSample.csv')
    file.registerTempTable("venmo")

    # convert to timestamps
    strings = sqlContext.sql("SELECT user1, user2, datetime FROM venmo")
    timestamp = strings.withColumn("datetime", unix_timestamp("datetime", "MM/dd/yyyy hh:mm:ss a")\
                            .cast("double").cast("timestamp"))

    # persist dataframe for efficiency
    timestamp.persist()

    # get date cutoff points
    dates = pd.date_range(start='7/1/2012', end='7/1/2016', freq='6MS')

    #calculate reciprocal relationships overtime (cumulative) 
    result = []
    for date in dates: 
        double_count = timestamp\
            .filter(timestamp.datetime < date)\
            .select(["user1", "user2"])\
            .distinct()
        relationships = double_count\
            .rdd.map(lambda x: frozenset({x[0], x[1]}))\
            .distinct().count()
        total = double_count.count()
        reciprocal = total-relationships
        try:
            result.append(reciprocal/relationships)
        except: 
            result.append(0)

    # calculate reciprocal relationships in whole network
    double_count = timestamp\
        .select(["user1", "user2"])\
        .distinct()
    relationships = double_count\
        .rdd.map(lambda x: frozenset({x[0], x[1]}))\
        .distinct().count()
    total = double_count.count()
    reciprocal = total-relationships
    answer = reciprocal/relationships
    print(answer)
    print(result)

    # write answer
    # f = open('greenwood_1.3.txt','w')
    # f.write("Total reciprocal relationships in Venmo network: ")
    # f.write(str(answer))
    # f.close()

    # unpersist dataframe from cache
    timestamp.unpersist()

    # plot results!
    labs = dates.strftime("%m/%y")
    final = pd.DataFrame(result, labs)
    final.columns = ["Proportion Reciprocal Transactions"]
    final.plot.bar(rot = 0)
    plt.title("Venmo Cumulative Reciprocal Transactions")
    plt.savefig("Reciprocal.png")

    # turn off drivers
    sc.stop()