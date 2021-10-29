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


def plot_series(time, series, format="-", start=0, end=None, color='blue'):
    plt.plot(time[start:end], series[start:end], format, color=color)
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

# Data Preparation - creating windowed tf.dataset.Datasets
dataset = tf.data.Dataset.from_tensor_slices(visits_train)
dataset = dataset.window(21, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(21))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(1).prefetch(1)

# for x, y in dataset:
#     print("x = ", x.numpy())
#     print("y = ", y.numpy())

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
layer0 = model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
layer1 = model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='causal', activation='relu'))
layer2 = model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, return_sequences=True)))
layer2 = model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30)))
layer4 = model.add(tf.keras.layers.Dense(1))
layer5 = model.add(tf.keras.layers.Lambda(lambda x: x * 1000))

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

# Model Callbacks

checkpoint_filepath = 'C:/Users/General Assembly/PycharmProjects/visitors_timeseries/model_checkpoint'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='mae',
    mode='min',
    save_best_only=True
)

reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.01, patience=5, verbose=1, min_delta=0.001, mode='min')

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=8, verbose=0, mode='min')

## SGD Optimizer - provided fairly bad results with losses around ~2000
# model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9))

## Adam Optimizer - significantly better results, losses in the high 100's
# model.compile(metrics=["mae"], optimizer=optimizer, loss=tf.keras.losses.Huber())

## RMSprop - best optimizer yet - low 700s
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='mae')

# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['mae'], loss=tf.keras.losses.Huber())
#
# model.fit(dataset, epochs=50, verbose=1, callbacks=[model_checkpoint, reduce_lr_loss, earlyStopping])

## PREDS

def get_preds(model, val_data):
    series = tf.expand_dims(val_data, axis=-1)
    tf_val_data = tf.data.Dataset.from_tensor_slices(series)
    tf_val_data = tf_val_data.window(21, shift=1, drop_remainder=False)
    tf_val_data = tf_val_data.flat_map(lambda x: x.batch(21))
    tf_val_data = tf_val_data.batch(1).prefetch(1)
    forecast = model.predict(tf_val_data)
    forecast = forecast * 10
    return forecast

tf.keras.backend.clear_session()
preds = get_preds(model, visits_test)

# ## PLOTTING

plt.figure(figsize=(10, 6))
plot_series(dates_test, visits_test)
plot_series(dates_test, preds, color='red')
plt.show()

# print(len(dates_test))
# print(len(visits_test))
# print(len(preds))