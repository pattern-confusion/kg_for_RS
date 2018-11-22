# indicate how many tasks can be split by spark cluster
# for heavy task increase this number might get a much better responding

SPARK_LIGHT_PARTITION = 4
SPARK_MEDIUM_PARTITION = 32
SPARK_HEAVY_PARTITION = 128
SPARK_DEFAULT_PARTITION = 4

LOCAL_SPARK_CONFIGURATION = {
    'spark.executor.memory': '32g',
    'spark.driver.memory': '32g',
    'spark.executor.cores': '8',
    'spark.num.executors': '1',
    'spark.python.worker.memory': '32g',
    'spark.driver.maxResultSize': '0'
}

SPARK_APPLICATION_NAME = 'Hestia 1.0'
SPARK_LOG_LEVEL = 'WARN'