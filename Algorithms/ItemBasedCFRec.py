"""
    This rec algorithm will use spark!
"""

from collections import defaultdict
from Kernel import RecModel, get_attr


def element_count(element_list):
    ret = defaultdict(int)
    for _ in element_list:
        ret[_] += 1
    return dict(ret)


def weighted_sim_jaccard(p1, p2):
    """
    带权 jaccard
    :param p1:
    :param p2:
    :return:
    """
    if len(p1) > len(p2):
        smin, smax = p2, p1
    else:
        smin, smax = p1, p2

    union = sum([value for key, value in smin.items()]) + sum([value for key, value in smax.items()])
    if union is 0: return 0

    common = 0
    for key in smin:
        if key in smax:
            common += min(smin[key], smax[key])
    return common / (union - common)


def top_matches(id, value, candidates, method, topk=20):
    matching = [(cid, method(value, cvalue)) for cid, cvalue in candidates.items() if cid != id]
    return sorted(matching, key=lambda x: x[1], reverse=True)[:topk]


class ItemCFAlgorithmTrainer_New(RecModel):
    """
        Original Item CF Offline training logic
            will output a item-item similarity matrix(in sparse matrix form).

        :param
            sim_method : one of 'Jaccard', 'cosine' to indicate a similarity calculation algorithm
            thresh     : 0.001 to indicate a threshold of minimum item relativity
            minRelatedItem : 0
            maxRelatedItem : 10000
            maxRelatedUser : 1000000
    """

    def __init__(self, params):
        self.params = params

    def fit(self, training_data):
        from Spark.SparkKernel import HestiaSparkResourceManager

        _user_min_bev = get_attr(self.params, 'minRelatedItem', 5)
        _user_max_bev = get_attr(self.params, 'maxRelatedItem', 200)
        _item_max_bev = get_attr(self.params, 'maxRelatedUser', 8000)
        _item_min_bev = get_attr(self.params, 'minRelatedUser', 20)
        _k_neighbors  = get_attr(self.params, 'k', 300)

        # filter invalid user behaviours
        valid_user_ids = set(training_data.map(
            lambda x: (x[0], 1)
        ).reduceByKey(
            lambda x, y: x + y
        ).filter(
            lambda x: _user_min_bev <= x[1] <= _user_max_bev
        ).map(
            lambda x: x[0]
        ).collect())

        training_data = training_data.filter(
            lambda x: x[0] in valid_user_ids
        )

        # transform record to item-dict
        item_act_records = training_data.map(
            lambda x: (x[1], [x[0]])
        ).reduceByKey(lambda x, y: x + y).map(
            lambda x: (x[0], element_count(x[1]))
        ).filter(
            # filter invalid item
            lambda x: _item_min_bev < len(x[1]) < _item_max_bev
        )

        item_act_records_dict = item_act_records.collectAsMap()
        item_act_records_brod = HestiaSparkResourceManager.hestiaSpark.broadcast(item_act_records_dict)

        similarity_matrix = item_act_records.map(
            lambda x: (x[0], top_matches(
                id=x[0],
                value=x[1],
                candidates=item_act_records_brod.value,
                method=weighted_sim_jaccard,
                topk=_k_neighbors,
            ))
        ).collectAsMap()

        HestiaSparkResourceManager.hestiaSpark.unbroadcast(item_act_records_brod)

        return similarity_matrix


class ItemCFRecommender():
    def recommend(self, user_act_history, model):
        """
            Online Recommend Logic

            :param
                user_act_history = user_act_history is a spark rdd, (user_id, user_history)

                where user_record is a tuple contains user action history (item_id, item_id, item_id)
                      item_similarity_matrix is item cf recommendation model (a map : {item_id1: [item_id2, similarity]})

            :return
                (item1, item2, item3) : a tuple of recommendation items.(sorted by relativity)
        """

        def __recommend_for_user(uid, user_acts, model):
            def zero_init():
                return 0.0

            user_acts = set(user_acts)

            ret = defaultdict(zero_init)

            for item in user_acts:

                if item in model:

                    for similarity_pair in model[item]:

                        sim_item_id = similarity_pair[0]
                        similarity = similarity_pair[1]

                        if sim_item_id not in user_acts:
                            ret[sim_item_id] += similarity

            raw_result = sorted([(iid, score) for iid, score in dict(ret).items()], key=lambda x: x[1], reverse=True)
            return raw_result[: 500]

        user_recommendation = user_act_history.map(
            lambda x: (x[0], __recommend_for_user(x[0], x[1], model))
        ).collectAsMap()

        return user_recommendation


class KGCFRecommender():
    def recommend(self, user_act_history, model):
        """
            Online Recommend Logic

            :param
                user_act_history = user_act_history is a spark rdd, (user_id, user_history)

                where user_record is a tuple contains user action history (item_id, item_id, item_id)
                      item_similarity_matrix is item cf recommendation model (a map : {item_id1: [item_id2, similarity]})

            :return
                (item1, item2, item3) : a tuple of recommendation items.(sorted by relativity)
        """

        def __recommend_for_user(uid, user_acts, model):
            def zero_init():
                return 0.0

            def format_rec(l):
                if l:
                    _max, _min = l[0][1], l[-1][1]
                    return {_id: (score-_min)/_max for _id, score in l}
                else:
                    return {}

            ret = defaultdict(zero_init)
            for item in user_acts:

                if item in model:

                    for similarity_pair in model[item]:

                        sim_item_id = similarity_pair[0]
                        similarity = similarity_pair[1]

                        ret[sim_item_id] += similarity

            raw_result = sorted([(iid, score) for iid, score in dict(ret).items()], key=lambda x: x[1],
                                reverse=True)
            return format_rec(raw_result[: 500])

        user_recommendation = user_act_history.map(
            lambda x: (x[0], __recommend_for_user(x[0], x[1], model))
        ).collectAsMap()

        return user_recommendation