from Kernel import RecModel
from collections import defaultdict

class HotRec(RecModel):

    def __zero_init(self):
        return 0

    def __init__(self):
        self.__forbidden_list = defaultdict(set)
        self.__rec_model = None

    def fit(self, training_data):

        print('Hot Rec model ready to fit data.')

        __movie_clicks = defaultdict(self.__zero_init)

        for _triple in training_data:
            # _triple is user-movie-rating entity
            self.__forbidden_list[_triple[0]].add(_triple[1])
            __movie_clicks[_triple[1]] = __movie_clicks[_triple[1]] + 1

        self.__rec_model = [(key, value) for key, value in __movie_clicks.items()]
        self.__rec_model.sort(key=lambda x: x[1], reverse=True)

        print('model trained successfully. with %d identity movie items' % len(self.__rec_model))

    def predict(self, predict_users):

        _user_history = self.__forbidden_list[predict_users]

        # a little trick here to accelerate calculation
        return list(filter(lambda x: x not in self.__forbidden_list[predict_users],
                           self.__rec_model[:len(_user_history) + 100]))
