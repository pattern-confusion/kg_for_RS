def cal_f1(pred, real, with_score=True):
    if len(pred) != len(real):
        raise Exception('unmatched data length! please recheck your input.')

    correct = 0
    predicted = 0
    interested = 0

    for recommendation, user_pref in zip(pred, real):

        interested += len(user_pref)
        predicted += len(recommendation)

        for rec in recommendation:
            if with_score: rec = rec[0]

            if rec in user_pref:
                correct += 1

    # laplace smooth
    precise = (correct + 1) / (predicted + 1)
    recall  = (correct + 1) / (interested + 1)
    f1 = (2 * precise * recall) / (precise + recall)

    return precise, recall, f1