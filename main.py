import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv('daily-website-visitors.csv')
df = df[['Date', 'Unique.Visits']]
df = df.rename(columns={'Unique.Visits': 'unique_visits', 'Date': 'date'})
df['date'] = pd.to_datetime(df.date)

df = df.replace(',', '', regex=True)
df['unique_visits'] = df['unique_visits'].astype(int)

## Converting full df to array
all_visits = np.array(df['unique_visits'])
all_dates = np.array(df['date'])

# ## Setting cutoff
# timestamp = pd.to_datetime('1/1/2019')
# train_df = df.loc[df.date <= timestamp]
# test_df = df.loc[df.date > timestamp]
#
# ## Train and Validation Split
# visits_train = np.array(train_df['unique_visits'])
# visits_test = np.array(test_df['unique_visits'])
# dates_train_array = np.array(train_df['date'])
# dates_test_array = np.array(test_df['date'])


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


# Train and Validation Split
split_time = 1571
visits_train = all_visits[:split_time]
visits_test = all_visits[split_time:]
dates_train = all_dates[:split_time]
dates_test = all_dates[split_time:]

naive_forecast = all_visits[split_time - 1:-1]

# Display plot of naive forecast
# plt.figure(figsize=(10, 6))
# plot_series(dates_test, visits_test, start=0, end=150)
# plot_series(dates_test, naive_forecast, start=1, end=151)
# plt.show()

# Finding our baseline accuracy with MAE - results in an MAE of 515
# print(keras.metrics.mean_absolute_error(visits_test, naive_forecast).numpy())

# Data Preparation - creating
dataset = tf.data.Dataset.from_tensor_slices(visits_train)
dataset = dataset.window(30, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(30))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(1).prefetch(1)

# for x, y in dataset:
#     print("x = ", x.numpy())
#     print("y = ", y.numpy())

model = tf.keras.models.Sequential()
layer0 = model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
layer1 = model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True))
layer2 = model.add(tf.keras.layers.SimpleRNN(30))
layer3 = model.add(tf.keras.layers.Dense(1))
layer4 = model.add(tf.keras.layers.Lambda(lambda x: x * 100))

# model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9))
model.compile(loss="mae", optimizer='adam')
model.fit(dataset, epochs=30, verbose=1)
