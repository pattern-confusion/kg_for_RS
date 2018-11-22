import pickle as pkl
import pickle
import numpy

def __zip2dic(zipped_list):
    return {uid: rec for uid, rec in zipped_list}

with open('rec', 'rb') as rec_file:
    recommendation = __zip2dic(pickle.load(rec_file))

with open('UserMovie_test2.txt', 'r') as user_file:
    user_list = user_file.readlines()

with open('final_rec.txt', 'w') as saving_file:
    for line in user_list:
        if int(line) in recommendation:
            _rec_list = recommendation[int(line)]

            _rec_list = _rec_list[: 50]

            for _rec_pair in _rec_list:
                saving_file.write(str(int(line)))
                saving_file.write('\t')
                saving_file.write(str(_rec_pair[0]))
                saving_file.write('\n')
