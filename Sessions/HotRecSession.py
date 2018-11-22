from Algorithms.HotRec import HotRec
from Kernel import RecSession, get_attr


class HotRecSession(RecSession):

    def __init__(self):
        pass

    def recommend(self, data, predict_user_list, params):

        _rec_n = get_attr(params, 'rec_length', 20)

        # transform data to 0-1 ratings
        _data = [(_[0], _[1], 1) for _ in data]

        # initialize rec model
        _model = HotRec()

        _model.fit(_data)

        return [_model.predict(uid)[: _rec_n] for uid in predict_user_list]