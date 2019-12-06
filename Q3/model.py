import tensorflow as tf
import numpy as np
from numpy.random import RandomState

import pandas as pd

train_df = pd.read_csv('./heart_train.csv')
test_df = pd.read_csv('./heart_test.csv')


# print (train_df.head())

# print (test_dataset.dtypes)

# converting famhist from categorical to numerical value
train_df['famhist'] = pd.Categorical(train_df['famhist'])
train_df['famhist'] = train_df.famhist.cat.codes

# print (train_df.head())

target = train_df.pop('chd')

dataset = tf.data.Dataset.from_tensor_slices((train_df.values, target.values))

# for feat, targ in dataset.take(5):
#   print ('Features: {}, Target: {}'.format(feat, targ))

tf.constant(train_df['famhist'])

# shuffle and batch - normalization
train_dataset = dataset.shuffle(len(train_df)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model
  
model = get_compiled_model()
model.fit(train_dataset, epochs=20)