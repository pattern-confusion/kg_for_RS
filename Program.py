"""
    This file implemented a disturbuted algorithm with spark 2.3
"""

try:
    import pyspark
    import numpy as np
    import pandas as pd
    import pickle as pkl
    from sklearn.model_selection import KFold
    from Kernel import *
except ImportError as e:
    print('Sorry you may not install necessary package.')


_rating_file_path = 'UserMovie_train.txt'
_movie_info_path = 'Movie.txt'
_sample_data_count = 2500000 # set this to be -1 to use all data.


def __load_data():

    # compress data into numerical format
    def __format_data(line):
        elements = line.split('\t')
        return int(elements[0]), int(elements[1]), float(elements[3].strip())

    with open(_rating_file_path, 'r', encoding='utf-8') as file:
        if _sample_data_count == -1:
            lines = file.readlines()[1:] # omit the first line.
        else:
            lines = [file.readline() for i in range(_sample_data_count)][1:]
    return np.array([__format_data(line) for line in lines]).astype(np.int)


def __load_movie_info():
    with open(_movie_info_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]  # omit the first line.

    return np.array([line.split('\t') for line in lines])


if __name__ == '__main__':
    _ratings = __load_data()
    _movies  = __load_movie_info()
    _final_gathering = False

    if _final_gathering:

        _training_data = _ratings

        _predict_data = generate_user_center_perf(_ratings).items()
        _predict_user_list = [key for key, value in _predict_data]
        _predict_user_pref = [set(value) for key, value in _predict_data]

        # rewrite here to test different rec model.

        from Sessions.UCFRecSession import UCFRecSession

        _rec_session = UCFRecSession()

        _recommendation = _rec_session.recommend(
            data=_training_data, predict_user_list=_predict_user_list, params={'rec_length': 50}
        )

        _recommendation = zip(_predict_user_list, _recommendation)

        with open('rec', 'wb') as file:
            pkl.dump(_recommendation, file)

    else:

        for _itr_idx, (_training_idxs, _test_idxs) in enumerate(KFold(n_splits=3, shuffle=False).split(X=_ratings)):

            _training_data = _ratings[_training_idxs]
            _test_data = generate_user_center_perf(_ratings[_test_idxs]).items()

            _predict_user_list = [key for key, value in _test_data]
            _predict_user_pref = [set(value) for key, value in _test_data]

            # rewrite here to test different rec model.

            from Sessions.UCFRecSession import UCFRecSession

            _rec_session = UCFRecSession()

            _recommendation = _rec_session.recommend(
                data=_training_data, predict_user_list=_predict_user_list, params={'rec_length': 50}
            )

            print('Recommendation generate successfully, with length %d' % len(_recommendation))

            from Evaluations import cal_f1

            precise, recall, f1 = cal_f1(pred=_recommendation, real=_predict_user_pref)

            print('Rec Algorithm test over at itr %d, with precise %.5f, recall %.5f, f1 %.5f' %
                  (_itr_idx + 1, precise, recall, f1))

else:
    print('Sorry this file must be invoke by operating system. do not import this.')