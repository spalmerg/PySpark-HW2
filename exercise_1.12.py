from pyspark import SparkContext
from pyspark.sql import HiveContext
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql.functions import *

def count(row):
    if row[1] is None:
        return len(set(row[3]))
    elif row[3] is None:
        return len(set(row[1]))
    else:
        return(len(set(row[1] + row[3])))

if __name__ == "__main__":

    # set up drivers
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = HiveContext(sc)

    # read in data
    file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
        .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/venmo/venmoSample.csv')
    file.registerTempTable("venmo")

    # query users
    df = sqlContext.sql("SELECT user1, user2 from venmo")

    # all connections for each user
    u1 = df.groupBy('user1').agg(collect_set('user2'))\
        .withColumnRenamed("collect_set(user2)", "Friends_1")
    u2 = df.groupBy('user2').agg(collect_set('user1'))\
        .withColumnRenamed("collect_set(user1)", "Friends_2")

    # undirected graph
    undirected = u1.join(u2, u1.user1 == u2.user2, "full").rdd
    undirected = undirected.map(count).collect()

    # outdegree
    outdegree = u1.map(lambda x: len(set(x[1]))).collect()

    # indegree
    indegree = u2.map(lambda x: len(set(x[1]))).collect()

    # plot undirected distribution
    plt.figure(0)
    plt.hist(undirected, range = [0,30])
    plt.xlabel("Connections")
    plt.title("Venmo Undirected Degree Distribution")
    plt.ylabel("Users")
    plt.savefig("Undirected.png")

    # plot outdegree
    plt.figure(1)
    plt.hist(outdegree, range = [0,30])
    plt.xlabel("Connections")
    plt.title("Venmo Out-Degree Distribution")
    plt.ylabel("Users")
    plt.savefig("Outdegree.png")

    # plot indegree
    plt.figure(2)
    plt.hist(indegree, range = [0,30])
    plt.xlabel("Connections")
    plt.title("Venmo In-Degree Distribution")
    plt.ylabel("Users")
    plt.savefig("Indegree.png")

    # turn off driver
    sc.stop()

