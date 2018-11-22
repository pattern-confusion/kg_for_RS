from Spark.SparkKernel import HestiaSparkResourceManager
from Spark.SparkConf import *
from Kernel import RecSession, get_attr
from Algorithms.CF_ALS import MFTrainer

# initialize spark context
hestia_spark = HestiaSparkResourceManager.get_HestiaSpark()

class SVD_ALS_RecSession(RecSession):
    def recommend(self, data, predict_user_list, params):
        _rec_n = get_attr(params, 'rec_length', 20)

        data = [(int(x[0]), int(x[1]), float(x[3])) for x in data]
        _data_rdd = hestia_spark.sparkContext.parallelize(data, numSlices=512)

        _model = MFTrainer(params=params)

        _model.fit(_data_rdd)

        ret = []
        for uid in predict_user_list:
            try:
                ret.append(_model.predict(uid)[: _rec_n])
            except Exception as e:
                print('Error happened when recommend with user %s ' % uid)
                print(e)

                ret.append([]) # append an empty list anyway.
        return ret