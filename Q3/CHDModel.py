from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.random import RandomState

import functools
import pandas as pd
import tensorflow as tf
import numpy as np

# ----- CODE BASED ON : https://www.tensorflow.org/tutorials/load_data/csv -----

LABEL_COLUMN = 'chd'
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, 
      label_name=LABEL_COLUMN,)
  return dataset

raw_train_data = get_dataset('./heart_train.csv')
raw_test_data = get_dataset('./heart_test.csv')

NUMERIC_FEATURES = ['sbp','tobacco', 'ldl','adiposity','typea','obesity','alcohol','age']

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

# Packing numeric data
packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

# Normalizing data
desc = pd.read_csv('./heart_train.csv')[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)

# Packing categorical data
CATEGORIES = {
    'famhist': ['Present', 'Absent'],
}
categorical_columns = []

for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)

# Assemble both into pre-processing layer
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

# Introducing variable learning rate for Adam optimizer
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=128*20,
  decay_rate=1,
  staircase=False)

# Train, Evaluate, and Predict
model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=30,verbose=2, steps_per_epoch=128)
test_loss, test_accuracy = model.evaluate(test_data, steps=128)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
predictions = model.predict(test_data, steps=64)

