from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark.sql.functions import *
import datetime
import re
from heapq import nlargest

def findEmojis(description):
    regex = r'[^\w\s,.\'!)(-/\x00\uFE0F]'
    emojis = re.findall(regex, description)
    return emojis


if __name__ == "__main__":

    # set up drivers
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

    # read in data and convert date to weekday
    file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
        .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/venmo/venmoSample.csv')
    file.registerTempTable("venmo")
    rdd = sqlContext.sql("SELECT date_format(datetime, 'E') AS DOW, description FROM venmo").rdd
    rdd.persist()

    most popular overall emojis
    emojis = rdd.flatMap(lambda x: findEmojis(x[1]))\
        .map(lambda x: (x,1))\
        .reduceByKey(lambda x, y: x+y)\
        .sortBy(lambda x: x[1], ascending = False)

    print("Top Emojis Overall")
    print(emojis.take(10))

    # most popular emojis by weekday
    emojis_DOW = rdd.map(lambda x: (x[0], findEmojis(x[1])))\
        .flatMapValues(lambda x: x)\
        .map(lambda x: (x, 1))\
        .reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[0][0], (x[0][1], x[1])))\
        .groupBy(lambda x: x[0])\
        .flatMap(lambda x: nlargest(5, list(x[1]), key = lambda x: x[1][1]))\
        .collect()

    print("Top Emojis By Day")
    print(emojis_DOW)

    rdd.unpersist()
    sc.stop()