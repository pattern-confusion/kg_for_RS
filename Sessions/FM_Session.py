from Algorithms.FM import FM_Rec
from Kernel import RecSession, get_attr, generate_user_center_perf
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np
import random

class FMRecSession(RecSession):

    def __init__(self):
        self.movie_id_set = set()
        self.user_id_set = set()
        self._user_prefs = None

    def load_cached_rec(self, relation_name):
        _saving_name = 'rec_{0}'.format(relation_name)
        with open(_saving_name, 'rb') as f:
            return pickle.load(f)

    def dump_cached_rec(self, relation_name, recommendation):
        _saving_name = 'rec_{0}'.format(relation_name)
        with open(_saving_name, 'wb') as f:
            return pickle.dump(recommendation, f)

    def __preprocessing_data(self, original_data, samples=8000000):
        _uids, _iids = list(self.user_id_set), list(self.movie_id_set)

        faked_data = []
        self._user_prefs = generate_user_center_perf(original_data)
        for i in range(samples):
            _ruid = _uids[random.randint(0, len(_uids) - 1)]
            _riid = _iids[random.randint(0, len(_iids) - 1)]

            while(_riid in self._user_prefs[_ruid]):
                _riid = _iids[random.randint(0, len(_iids))]

            faked_data.append((_ruid, _riid))

        return faked_data

    def __generate_predict_data(self, uid):
        predict_data = []
        for mid in self.movie_id_set:
            if mid not in self._user_prefs[uid]:
                predict_data.append((uid, mid))
        return predict_data

    def recommend(self, data, predict_user_list, params):

        _rec_n = get_attr(params, 'rec_length', 20)


        # transform data to 0-1 ratings
        _data = [(_[0], _[1]) for _ in data]
        for _uid, _iid in _data:
            self.user_id_set.add(_uid)
            self.movie_id_set.add(_iid)

        _faked_data = self.__preprocessing_data(_data, samples=len(_data))

        _y = np.concatenate((np.repeat(1, len(_data)), np.repeat(0, len(_faked_data))))

        _encoder = OneHotEncoder()
        _train_data = _encoder.fit_transform(X=_data)

        model = FM_Rec()
        model.fit(_train_data, y=_y)

        recommendations = {}
        for uid in self.user_id_set:

            print('1')

            _predict_data = self.__generate_predict_data(uid)
            _mids = [mid for uid, mid in _predict_data]
            _rec_score = [_[1] for _ in model.predict(_encoder.transform(_predict_data))]
            _recommendation = sorted(zip(_mids, _rec_score), key=lambda x: x[1], reverse=True)[: 500]
            recommendations[uid] = _recommendation

        return [recommendations[uid][: _rec_n] if uid in recommendations else []
            for uid in predict_user_list]

