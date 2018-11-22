"""
    This file defined behaviour of matrix factorize algorithms
    Algorithm supported now:
        ALS -- update on 3.25, 2018
    @author: Phoenix.z 2007-2018
"""

from Kernel import RecModel, get_attr
from pyspark.mllib.recommendation import ALS

"""
    Spark ALS Model Trainer
        will output a MF Model

    :param
        rank    : num of factors
        itr_times : training iteration times
        regParam  : regularize param
        NMF       : a param indicate if or not do non negative MF
"""


class MFTrainer(RecModel):

    def __init__(self, params):
        self.params = params
        self.recommendation = None

    def fit(self, training_data):
        _rec_num = get_attr(self.params, 'rec_length', 20)
        _rank = get_attr(self.params, 'rank', 50)
        _reg  = get_attr(self.params, 'regParam', 0.5)
        _iteration = get_attr(self.params, 'iter', 5)
        _model = ALS.train(training_data, rank=_rank, nonnegative=True, iterations=_iteration)
        self.recommendation = _model.recommendProductsForUsers(_rec_num).collectAsMap()

    def predict(self, predict_users):
        return [(_.product, _.rating) for _ in self.recommendation[predict_users]] if predict_users in self.recommendation else []
