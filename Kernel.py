from collections import defaultdict


class RecModel():
    def __init__(self, params=None):
        self.params = params
        self.name='Rec model'

    def fit(self, training_data):
        pass

    def predict(self, predict_users):
        pass


class RecSession():
    def __init__(self):
        self.name = 'Rec Session'

    def recommend(self, data, predict_user_list, params):
        pass


def get_attr(dictionary, attr, default_value):
    if isinstance(dictionary, dict) and attr in dictionary:
        return dictionary[attr]
    else: return default_value


def generate_user_center_perf(triples):

    user_center_pref = defaultdict(list)

    for _triple in triples:
        user_center_pref[_triple[0]].append(_triple[1])

    return dict(user_center_pref)


def generate_item_center_perf(triples):
    item_center_pref = defaultdict(list)

    for _triple in triples:
        item_center_pref[_triple[1]].append(_triple[0])

    return dict(item_center_pref)