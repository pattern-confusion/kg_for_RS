from tffm import TFFMClassifier
from Kernel import *
import tensorflow as tf


class FM_Rec(RecModel):

    def __init__(self):
        self.model = None

    def fit(self, training_data, y):
        self.model = TFFMClassifier(
            order=2,
            rank=64,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
            n_epochs=100,
            batch_size=-1,
            init_std=0.001,
            input_type='sparse',
            verbose=2
        )
        self.model.fit(X=training_data, y=y)

    def predict(self, predict_users):
        return self.model.predict_proba(predict_users)
