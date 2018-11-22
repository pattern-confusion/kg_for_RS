"""
     _   _ _____ ____ _____ ___    _
    | | | | ____/ ___|_   _|_ _|  / \
    | |_| |  _| \___ \ | |  | |  / _ \
    |  _  | |___ ___) || |  | | / ___ \
    |_| |_|_____|____/ |_| |___/_/   \_\

    Hestia is a Distributed Recommendation System

    latest version is 1.0 -- update on 7.20 2018 by Phoenix.z

    This file defines Hestia system's kernel behavior and data structure.

    Do not modify this file if not necessary, it cause global effect towards whole system.
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from Spark.SparkConf import *
import pyspark.broadcast
import pyspark.rdd


# -*- coding: utf-8 -*-

class HestiaSparkResourceManager:
    """
        This is Hestia Spark resource manager
            This class is used to help Hestia applications to use spark function
            notice that it is Hestia kernel function, please do not modify this if not necessary.
    """

    def __init__(self, conf=None, cluster_mode=False):
        spark_configuration = SparkConf()
        if isinstance(conf, dict):
            for key, value in conf.items():
                spark_configuration.set(key, value)
        else:
            print('No spark setting detected, will use default spark configuration')
            conf = LOCAL_SPARK_CONFIGURATION
            for key, value in conf.items():
                spark_configuration.set(key, value)

        if cluster_mode:
            self.sparkContext = SparkContext(
                master='yarn',
                appName=SPARK_APPLICATION_NAME,
                conf=spark_configuration
            )
            self.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
            self.sparkContext.addPyFile('Hestia.zip')
            self.sparkSession = SparkSession.builder.master('yarn').getOrCreate()
        else:
            self.sparkContext = SparkContext(
                master='local[*]',
                appName=SPARK_APPLICATION_NAME,
                conf=spark_configuration
            )
            self.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
            self.sparkSession = SparkSession.builder.master('local[*]').getOrCreate()

        print('Hestia Spark Environment is running.')

        self.broadcasted = set()
        self.persisted = set()

    def setStorageLevel(self, changeTo):
        """
        :param changeTo:
        :return:
        """
        pass

    def broadcast(self, value):
        """
        broadcast a python variable to spark worker,
        detailed information can be found here:
            http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.Broadcast

        notice that all broadcasted variable should be unbroadcast(unpersist) manually.
        otherwise it will cause memory leak.

        :param value: a python variable that need to be broadcast
        :return: a spark broadcast instance
        """

        broadcast = self.sparkContext.broadcast(value)
        self.broadcasted.add(broadcast)

        return broadcast

    def unbroadcast(self, broadcasted):
        """
        Unpersist a spark broadcast object.
            if input variable is not a spark broadcast instance, will raise exception
        :param broadcasted: spark broadcast instance
        :return: no thing to return
        """
        if isinstance(broadcasted, pyspark.broadcast.Broadcast):

            broadcasted.unpersist()

            self.broadcasted.remove(broadcasted)

        else:

            raise Exception('only broadcasted value can be unpersist via this function.')

    def persist(self, rdd):
        """
        tell spark to store your rdd into memory temporarily
            if input variable is not a spark rdd instance, will raise exception
        :param value: a spark rdd that need to be parallelize
        :return: nothing to return
        """
        if isinstance(rdd, pyspark.rdd.RDD):

            self.persisted.add(rdd)

            rdd.persist()

        else:

            raise Exception('only rdd can be persist via this function.')

    def unpersist(self, rdd):
        """
        tell spark to store your rdd into memory temporarily
            if input variable is not a spark rdd instance, will raise exception
        :param value: a spark rdd that need to be parallelize
        :return: nothing to return
        """
        if isinstance(rdd, pyspark.rdd.RDD):

            rdd.unpersist()

            self.persisted.remove(rdd)

        else:

            raise Exception('only rdd can be unpersist via this function.')

    def clearAll(self):
        """
        clear all persisted value in spark, including broadcast and persist value.
            Notice this function should be invoke by Hestia kernel automatically
            And should not be modified if not necessary.

        for programmers: you can use this function to free all persisted value in spark.
        but it is also recommend you to free them manually, which might have higher performance.

        Notice calling this function can cause some exception and you may want to catch them.

        :return: nothing to return
        """

        for v in self.broadcasted:
            if isinstance(v, pyspark.broadcast.Broadcast):
                v.unpersist()

        for v in self.persisted:
            if isinstance(v, pyspark.rdd.RDD):
                v.unpersist()

        self.broadcasted.clear()
        self.persisted.clear()

    hestiaSpark = None
    spark_context = None
    spark_runtime = None
    sqlContext = None

    @classmethod
    def get_HestiaSpark(cls, spark_conf=None, cluster_mode=False):
        if cls.hestiaSpark is None:
            cls.hestiaSpark = HestiaSparkResourceManager(conf=spark_conf, cluster_mode=cluster_mode)
            cls.spark_context = cls.hestiaSpark.sparkContext
            cls.spark_runtime = cls.hestiaSpark.sparkSession
            cls.sqlContext = cls.spark_runtime._wrapped
        return cls.hestiaSpark
