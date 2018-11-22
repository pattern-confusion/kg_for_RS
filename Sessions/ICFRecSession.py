from Spark.SparkKernel import HestiaSparkResourceManager
from Spark.SparkConf import *
from Kernel import RecSession, get_attr
from Algorithms.ItemBasedCFRec import ItemCFAlgorithmTrainer_New, ItemCFRecommender

# initialize spark context
hestia_spark = HestiaSparkResourceManager.get_HestiaSpark()

class ICFRecSession(RecSession):
    def recommend(self, data, predict_user_list, params):
        _rec_n = get_attr(params, 'rec_length', 20)

        data = [(int(x[0]), int(x[1])) for x in data]
        _data_rdd = hestia_spark.sparkContext.parallelize(data, numSlices=SPARK_DEFAULT_PARTITION)

        _model = ItemCFAlgorithmTrainer_New(params=params)

        _sim_matrix = _model.fit(_data_rdd)

        _recommendation = ItemCFRecommender().recommend(_data_rdd.map(
            lambda x: (x[0], [x[1]])
        ).reduceByKey(lambda x, y: x + y), model=_sim_matrix)

        return [_recommendation[uid][: _rec_n] if uid in _recommendation else [] for uid in predict_user_list]

