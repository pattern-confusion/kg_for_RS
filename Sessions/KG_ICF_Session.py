from Spark.SparkKernel import HestiaSparkResourceManager
from Spark.SparkConf import *
from Kernel import RecSession, get_attr
from Algorithms.UserBasedCFRec import UserCFAlgorithmTrainer, UserCFRecommender
from Algorithms.ItemBasedCFRec import ItemCFAlgorithmTrainer_New, ItemCFRecommender, KGCFRecommender
from collections import defaultdict
import pickle

# initialize spark context
hestia_spark = HestiaSparkResourceManager.get_HestiaSpark()


def str_format(s):
    if not isinstance(s, str):
        s = str(s).strip()
    if '(' in s:
        return s[s.index('(') + 1: -2].strip()
    else:
        return s.strip()


def data_format(d):
    _vaild_year = [str(i) for i in range(1940, 2019)]
    for _year in _vaild_year:
        if _year in d:
            return _year
    return None


_info_tag_map = {
    2: '导演',
    3: '编剧',
    4: '主演',
    5: '类别',
    6: '国家',
    7: '时长',
    8: '出品时间',
    9: '标签'
}


class KGRecSession(RecSession):

    def __init__(self):
        self._merge_weight = {'国家': 1.0, '出品时间': 1.0, '导演': 1.0, '主演': 1.0}
        self._processing_relations = {'国家': str_format, '出品时间': data_format, '主演': str_format, '导演': str_format, }
        self._relation_forwards_projections = {key: defaultdict(set) for key in self._processing_relations}
        self._relation_backwards_projections = {key: defaultdict(set) for key in self._processing_relations}

    def load_cached_rec(self, relation_name):
        _saving_name = 'rec_{0}'.format(relation_name)
        with open(_saving_name, 'rb') as f:
            return pickle.load(f)

    def dump_cached_rec(self, relation_name, recommendation):
        _saving_name = 'rec_{0}'.format(relation_name)
        with open(_saving_name, 'wb') as f:
            return pickle.dump(recommendation, f)

    def __format_movies_info(self, movie_content):
        for line in movie_content:
            _elements = line.split('\t')

            for _relation_idx, _relation in _info_tag_map.items():

                if _relation in self._processing_relations:
                    _formatter = self._processing_relations[_relation]

                    _head, _tails = int(_elements[0].strip()), _elements[_relation_idx].strip().split(';')

                    for _tail in _tails:
                        tail = _formatter(_tail)

                        if tail:
                            self._relation_forwards_projections[_relation][_head].add(tail)
                            self._relation_backwards_projections[_relation][tail].add(_head)

    def __ranking(self, _recommendations):
        _base_sroce = _recommendations['base']
        for user in _base_sroce:
            rec_pair = _base_sroce[user]

            for idx in range(len(rec_pair)):

                rec_pair[idx] = list(rec_pair[idx])
                _rec_item = rec_pair[idx][0]

                for _relation in self._processing_relations:

                    rec_pair[idx].append(0.0)

                    for _projection in self._relation_forwards_projections[_relation][_rec_item]:

                        if user in _recommendations[_relation] and _projection in _recommendations[_relation][user]:

                            rec_pair[idx][-1] += _recommendations[_relation][user][_projection]

        return _base_sroce

    def __bagging(self, UCF, ICF):

        _bagging_result = {}

        for uid, ICF_pairs in ICF.items():

            _new_pairs = defaultdict(float)

            UCF_pairs = UCF[uid]

            for rid, score in ICF_pairs + UCF_pairs:

                _new_pairs[rid] += score

            _bagging_result[uid] = [(rid, score) for rid, score in _new_pairs.items()]

        return _bagging_result

    def recommend(self, data, predict_user_list, params):

        _rec_n = get_attr(params, 'rec_length', 50)
        _movie_content = get_attr(params, 'movie_info', '')

        # begin to run icf on kg.
        self.__format_movies_info(_movie_content)
        '''
        for _relation in self._processing_relations:

            _forward_projection = self._relation_forwards_projections[_relation]

            _data = []
            for row in data:
                for _ in _forward_projection[row[1]]:
                    _data.append((row[0], _))

            _data_rdd = hestia_spark.sparkContext.parallelize(_data, numSlices=SPARK_DEFAULT_PARTITION)

            _model = ItemCFAlgorithmTrainer_New(params=params)

            _sim_matrix = _model.fit(_data_rdd)

            _recommendation = KGCFRecommender().recommend(_data_rdd.map(
                lambda x: (x[0], [x[1]])
            ).reduceByKey(lambda x, y: x + y), model=_sim_matrix)

            self.dump_cached_rec(_relation, _recommendation)
            
        self.dump_cached_rec('projection', self._relation_forwards_projections)
        data = [(int(x[0]), int(x[1])) for x in data]
        _data_rdd = hestia_spark.sparkContext.parallelize(data, numSlices=SPARK_DEFAULT_PARTITION)

        _model = ItemCFAlgorithmTrainer_New(params=params)

        _sim_matrix = _model.fit(_data_rdd)

        _recommendation = ItemCFRecommender().recommend(_data_rdd.map(
            lambda x: (x[0], [x[1]])
        ).reduceByKey(lambda x, y: x + y), model=_sim_matrix)

        self.dump_cached_rec('ICF', _recommendation)
        data = [(int(x[0]), int(x[1])) for x in data]
        _data_rdd = hestia_spark.sparkContext.parallelize(data, numSlices=SPARK_DEFAULT_PARTITION)

        _model = UserCFAlgorithmTrainer(params=params)

        _sim_matrix = _model.fit(_data_rdd)

        _recommendation = UserCFRecommender().recommend(_data_rdd.map(
            lambda x: (x[0], [x[1]])
        ).reduceByKey(lambda x, y: x + y), model=_sim_matrix)

        self.dump_cached_rec('UCF', _recommendation)
        '''

        # final gathering stage
        self._relation_forwards_projections = self.load_cached_rec('projection')
        _recommendations = {_relation : self.load_cached_rec(_relation) for _relation in self._processing_relations}
        _recommendations['base'] = self.__bagging(self.load_cached_rec('ICF'), self.load_cached_rec('UCF'))
        _recommendation = self.__ranking(_recommendations)

        self.dump_cached_rec('rec_final_gathering', _recommendation)

        _recommendation = self.load_cached_rec('rec_final_gathering')
        for uid in _recommendation:

            for rec_pair in _recommendation[uid]:

                rec_pair[1] += rec_pair[2] * 0.04 + rec_pair[3] * 0.04 + rec_pair[4] * 0.04 + rec_pair[5] * 0.04

            _recommendation[uid] = sorted(_recommendation[uid], key=lambda x: x[1], reverse=True)

        return [_recommendation[uid][: _rec_n] if uid in _recommendation else [] for uid in predict_user_list]
