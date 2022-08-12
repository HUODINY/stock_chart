import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import tensorflow as tf
import koreanize_matplotlib 
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
df = usdkrw = fdr.DataReader('USD/KRW', '2015') # 달러당 원화
st.table(df)
df = df[['Close']].copy()

# minmaxscaler 데이터 정규화 
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df[['Close_mms']] = mms.fit_transform(df[['Close']])

# 데이터 셋 분리

# 특정 시점 기준으로 데이터 나누기
date = '2022-05-01'
train = df.loc[df.index < date, 'Close_mms']
test = df.loc[date <= df.index, 'Close_mms']

# window dataset
ds = tf.data.Dataset.range(10) 
ds = ds.window(5, shift=1, drop_remainder=False)

ds = tf.data.Dataset.range(10) 
ds = ds.window(5, shift=1, drop_remainder=True)

def windowed_dataset(series, window_size, batch_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1],w[-1]))
    return ds.batch(batch_size).prefetch(1)

# 루프에서 많은 모델을 생성하는 경우이 전역 상태는 시간이 지남에 따라 증가하는 메모리를 소비하므로
# 세션을 지워 이전 모델 및 레이어의 혼란을 방지하는데 도움이 됩니다.
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# 하이퍼파라미터를 설정합니다.
window_size = 20
batch_size = 32

# 데이터 집합을 적절하게 만들기 위해 windowed_dataset 함수를 사용합니다.
train_set = windowed_dataset(train, window_size, batch_size)
test_set = windowed_dataset(test, window_size, batch_size)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, LSTM, Bidirectional, Lambda, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error

# 모델층 구성
model = Sequential([Conv1D(filters=32, kernel_size=5, strides=1, activation='relu',
                          input_shape=[None, 1]),
                   LSTM(32, return_sequences=True, dropout=0.2),
                   LSTM(16),
                   Dense(1)
                   ])

# 컴파일
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: 1e-8 * 10**(epoch/20))
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.9)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=10)

model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])

# 학습
history = model.fit(train_set, validation_data=test_set, epochs=100, callbacks=[early_stop, lr_schedule])

# 학습률과 손실
df_hist = pd.DataFrame(history.history)

# 세션 삭제 및 모델 재구성
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# 모델의 층구성
model = Sequential([Conv1D(filters=32, kernel_size=5, strides=1, activation='relu',
                          input_shape=[None, 1]),
                   LSTM(32, return_sequences=True, dropout=0.2),
                   LSTM(16),
                   Dense(1)
                   ])

# 컴파일
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5, momentum=0.9)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=10)

model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])

# 모델 학습
history = model.fit(train_set, epochs=100, callbacks=[early_stop])

# 예측
df_hist = pd.DataFrame(history.history)

window_size
batch_size
def window_ds(series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w : w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

split_time = train.shape[0]
data_ds = window_ds(df['Close_mms'][:, np.newaxis], window_size, batch_size)
forecast = model.predict(data_ds)
y_predict = forecast[split_time - window_size:-1, -1]

# rmse
test_inverse = mms.inverse_transform(np.array(test).reshape(-1, 1))
y_predict_inverse = mms.inverse_transform(y_predict.reshape(-1, 1))
rmse = np.sqrt(np.mean(np.square(test_inverse - y_predict_inverse)))
