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
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from string import punctuation


def emojiCount(description):
    regex = r'[^\w\s,.\'!)(-/\x00\uFE0F]'
    emojis = re.findall(regex, description)
    return len(emojis)

def wordCount(description):
    regex = r'\w+'
    words = re.findall(regex, description)
    return len(words)

def punctCount(description):
    regex = r"[" + punctuation + "]"
    puncts = re.findall(regex, description)
    return len(puncts)

def vowelCount(description):
    regex = r'[aeiouy]'
    vowels = re.findall(regex, description)
    return len(vowels)

if __name__ == "__main__":

    # set up drivers
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

    # read in data
    file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
        .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/venmo/venmoSample.csv')
    file.registerTempTable("venmo")

    # query table and 
    df = sqlContext.sql("SELECT description FROM venmo")

    # convert to RDD for feature creation
    rdd = df.rdd

    # create features
    # month, punctcount, wordcount, emojicount, (word/word + emoji)
    features = rdd.map(lambda x: ((punctCount(x[0]), vowelCount(x[0]), wordCount(x[0]), emojiCount(x[0]))))\
       .map(lambda x: (x[0], x[1], x[2], x[3], x[3]/(x[2] + x[3] + 1)))

    # scale data for clustering (comensorate units)
    scaler = StandardScaler(withMean=True, withStd=True).fit(features)
    final = scaler.transform(features)\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[4]))

    #K-Means Clustering
    clusters = KMeans.train(final, 3, maxIterations=10, runs=10, initializationMode="random")

    # gather centroids
    centroids = pd.DataFrame(clusters.centers, columns = ["PunctCount", "VowelCount", "WordCount", "EmojiCount", "Prop"]).reset_index()
    centroids = centroids.rename(columns={"index":"Cluster"})
    centroids.to_csv("exercise2.3_centroids.csv")

    # plot centroids
    plt.figure(0)
    plt.bar(centroids["Cluster"], centroids["PunctCount"])
    plt.xlabel("Cluster")
    plt.title("Cluster Centroids: Punctuation (Standardized)")
    plt.ylabel("Punct")
    plt.savefig("exercise2.3_Punctuation.png")

    plt.figure(1)
    plt.bar(centroids["Cluster"], centroids["VowelCount"])
    plt.xlabel("Cluster")
    plt.title("Cluster Centroids: VowelCount (Standardized)")
    plt.ylabel("VowelCount")
    plt.savefig("exercise2.3_Vowel.png")

    plt.figure(2)
    plt.bar(centroids["Cluster"], centroids["WordCount"])
    plt.xlabel("Cluster")
    plt.title("Cluster Centroids: WordCount (Standardized)")
    plt.ylabel("WordCount")
    plt.savefig("exercise2.3_WordCount.png")

    plt.figure(3)
    plt.bar(centroids["Cluster"], centroids["EmojiCount"])
    plt.xlabel("Cluster")
    plt.title("Cluster Centroids: EmojiCount (Standardized)")
    plt.ylabel("EmojiCount")
    plt.savefig("exercise2.3_EmojiCount.png")

    plt.figure(4)
    plt.bar(centroids["Cluster"], centroids["Prop"])
    plt.xlabel("Cluster")
    plt.title("Cluster Centroids: Proportion Emoji (Standardized)")
    plt.ylabel("Prop Emoji")
    plt.savefig("exercise2.3_PropEmoji.png")

    # close driver
    sc.stop()