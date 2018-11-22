"""
    This rec algorithm will use spark!
"""

from collections import defaultdict
from Kernel import RecModel, get_attr


def sim_jaccard(p1, p2):
    if len(p1) > len(p2):
        smin, smax = p2, p1
    else:
        smin, smax = p1, p2

    union = len(smin) + len(smax)
    if union is 0: return 1

    common = 0
    for key in smin:
        if key in smax:
            common += 1
    return common / (union - common)


def top_matches(id, value, candidates, method, topk=20):
    matching = [(cid, method(value, cvalue)) for cid, cvalue in candidates.items() if cid != id]
    return sorted(matching, key=lambda x: x[1], reverse=True)[:topk]


class UserCFAlgorithmTrainer(RecModel):
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
        self.user_history = None

    def fit(self, training_data):
        from Spark.SparkKernel import HestiaSparkResourceManager

        _item_max_bev = get_attr(self.params, 'maxRelatedUser', 8000)
        _item_min_bev = get_attr(self.params, 'minRelatedUser', 5)
        _k_neighbors  = get_attr(self.params, 'k', 100)

        # filter invalid item behaviours
        valid_item_ids = set(training_data.map(
            lambda x: (x[1], 1)
        ).reduceByKey(
            lambda x, y: x + y
        ).filter(
            lambda x: _item_min_bev <= x[1] <= _item_max_bev
        ).map(
            lambda x: x[0]
        ).collect())

        training_data = training_data.filter(
            lambda x: x[1] in valid_item_ids
        )

        # transform record to user-dict
        user_act_records = training_data.map(
            lambda x: (x[0], [x[1]])
        ).reduceByKey(lambda x, y: x + y).map(
            lambda x: (x[0], set(x[1]))
        )

        user_act_records_dict = user_act_records.collectAsMap()
        user_act_records_brod = HestiaSparkResourceManager.hestiaSpark.broadcast(user_act_records_dict)

        similarity_matrix = user_act_records.map(
            lambda x: (x[0], top_matches(
                id=x[0],
                value=x[1],
                candidates=user_act_records_brod.value,
                method=sim_jaccard,
                topk=_k_neighbors,
            ))
        ).collectAsMap()

        HestiaSparkResourceManager.hestiaSpark.unbroadcast(user_act_records_brod)
        return similarity_matrix


class UserCFRecommender():
    def recommend(self, user_act_history, model):

        from Spark.SparkKernel import HestiaSparkResourceManager
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

            ret = defaultdict(zero_init)

            if uid in model:

                for sim_user_id, similarity in model[uid]:

                    if sim_user_id in user_acts:

                        sim_user_beh = user_acts[sim_user_id]

                        for beh in sim_user_beh:

                            if beh not in user_acts[uid]:

                                ret[beh] += similarity
            else:
                return []

            raw_result = sorted([(iid, score) for iid, score in dict(ret).items()], key=lambda x: x[1], reverse=True)
            return raw_result[: 500]

        user_act_records_dict = user_act_history.collectAsMap()
        user_act_records_brod = HestiaSparkResourceManager.hestiaSpark.broadcast(user_act_records_dict)

        user_recommendation = user_act_history.map(
            lambda x: (x[0], __recommend_for_user(x[0], user_act_records_brod.value, model))
        ).collectAsMap()

        HestiaSparkResourceManager.hestiaSpark.unbroadcast(user_act_records_brod)

        return user_recommendation
