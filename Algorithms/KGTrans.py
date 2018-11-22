"""
    Knowledge graph embedding methods.
"""

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np


class TransE:

    def __init__(self):
        self.model = None
        self._relations_df = None
        self._entity_count = 0
        self._relation_count = 0
        self._training_step = 100000
        self._embedding_size = 256
        self._batch_size = 128
        self._batch_idx = 0

    def generate_batch(self):
        if self._batch_idx + self._batch_size >= len(self._relations_df):
            self._batch_idx = 0
            self._relations_df = self._relations_df.sample(frac = 1.0)

        _ret = self._relations_df.iloc[self._batch_idx: self._batch_idx + self._batch_size]
        self._batch_idx += self._batch_size
        return _ret

    def data_preprocessing(self):
        df = pd.read_csv('../KG.txt', sep='\t')

        valid_tags = {'导演', '编剧', '国家', '主演', '类别'}
        df = pd.concat([df[df['relationship'] == tag] for tag in valid_tags])
        df['object'] = df['object'].apply(lambda x: str(x))
        df['subject'] = df['subject'].apply(lambda x: str(x))

        entities = pd.concat([df['object'], df['subject']])
        relations = df['relationship']

        entities_encoder = LabelEncoder()
        entities_encoder.fit(entities)
        relations_encoder = LabelEncoder()
        relations_encoder.fit(relations)

        df['object'] = entities_encoder.transform(df['object'])
        df['subject'] = entities_encoder.transform(df['subject'])
        df['relationship'] = relations_encoder.transform(df['relationship'])

        if isinstance(df, pd.DataFrame):
            self._relations_df = df.astype(dtype='int32')
            self._entity_count = max(df['subject'].max(), df['object'].max())
            self._relation_count = df['relationship'].max()

    def fit(self):
        input_entities, input_relations, output_entities = tf.placeholder(shape=[self._batch_size], dtype=tf.int32), \
                                                           tf.placeholder(shape=[self._batch_size], dtype=tf.int32), \
                                                           tf.placeholder(shape=[self._batch_size], dtype=tf.int32)

        enetity_embeddings = tf.Variable(tf.random_normal(
            shape=[self._entity_count, self._embedding_size], stddev=1.0
        ))

        relation_embeddings = tf.Variable(tf.random_normal(
            shape=[self._relation_count, self._embedding_size], stddev=1.0
        ))

        pred = tf.gather(enetity_embeddings, input_entities) + \
               tf.gather(relation_embeddings, input_relations)
        real = tf.gather(enetity_embeddings, output_entities)

        loss = tf.reduce_mean(tf.square(pred - real))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self._training_step):

                training_batch = self.generate_batch()
                _ = np.array(training_batch['object']).astype(np.int32).reshape(-1)

                sess.run(train_step, feed_dict={
                    input_entities: training_batch['object'],
                    input_relations: training_batch['relationship'],
                    output_entities: training_batch['subject']
                })

                if i % 1000 == 0:

                    print('Training at %d, loss: %.5f' % (i, sess.run(loss, feed_dict={
                        input_entities: training_batch['object'],
                        input_relations: training_batch['relationship'],
                        output_entities: training_batch['subject']
                    })))


class TransR:

    def __init__(self):
        self.model = None
        self._relations_df = None
        self._entity_count = 0
        self._relation_count = 0
        self._training_step = 10000
        self._embedding_size = 128
        self._relation_space_size = 256
        self._batch_size = 128
        self._batch_idx = 0
        self._onusing_relation = 0

    def generate_batch(self):
        # more work should be done here.

        self._onusing_relation += 1
        self._onusing_relation %= self._relation_count

        _onusing_df = self._relations_df[self._relations_df['relationship'] == self._onusing_relation]
        return _onusing_df.sample(n=self._batch_size)

    def data_preprocessing(self):
        df = pd.read_csv('../KG.txt', sep='\t')

        valid_tags = {'导演', '编剧', '国家', '主演', '类别'}
        df = pd.concat([df[df['relationship'] == tag] for tag in valid_tags])
        df['object'] = df['object'].apply(lambda x: str(x))
        df['subject'] = df['subject'].apply(lambda x: str(x))

        entities = pd.concat([df['object'], df['subject']])
        relations = df['relationship']

        entities_encoder = LabelEncoder()
        entities_encoder.fit(entities)
        relations_encoder = LabelEncoder()
        relations_encoder.fit(relations)

        df['object'] = entities_encoder.transform(df['object'])
        df['subject'] = entities_encoder.transform(df['subject'])
        df['relationship'] = relations_encoder.transform(df['relationship'])

        if isinstance(df, pd.DataFrame):
            self._relations_df = df.astype(dtype='int32')
            self._entity_count = max(df['subject'].max(), df['object'].max())
            self._relation_count = df['relationship'].max()

    def fit(self):
        input_entities, input_relations, output_entities = tf.placeholder(shape=[self._batch_size], dtype=tf.int32), \
                                                           tf.placeholder(shape=[self._batch_size], dtype=tf.int32), \
                                                           tf.placeholder(shape=[self._batch_size], dtype=tf.int32)

        enetity_embeddings = tf.Variable(tf.random_normal(
            shape=[self._entity_count, self._embedding_size], stddev=1.0
        ))

        relation_embeddings = tf.Variable(tf.random_normal(
            shape=[self._relation_count, self._relation_space_size], stddev=1.0
        ))

        relation_projection_matrix = tf.Variable(tf.random_normal(
            shape=[self._relation_count, self._embedding_size, self._relation_space_size], stddev=1.0
        ))

        _projection_matrix = relation_projection_matrix[self._onusing_relation]

        pred = tf.matmul(tf.gather(enetity_embeddings, input_entities), _projection_matrix) + \
               tf.gather(relation_embeddings, input_relations)

        real = tf.matmul(tf.gather(enetity_embeddings, output_entities), _projection_matrix)

        loss = tf.reduce_mean(tf.square(pred - real))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self._training_step):

                training_batch = self.generate_batch()
                _ = np.array(training_batch['object']).astype(np.int32).reshape(-1)

                sess.run(train_step, feed_dict={
                    input_entities: training_batch['object'],
                    input_relations: training_batch['relationship'],
                    output_entities: training_batch['subject']
                })

                if i % 1000 == 0:

                    print('Training at %d, loss: %.5f' % (i, sess.run(loss, feed_dict={
                        input_entities: training_batch['object'],
                        input_relations: training_batch['relationship'],
                        output_entities: training_batch['subject']
                    })))


trans_model = TransE()
trans_model.data_preprocessing()
trans_model.fit()