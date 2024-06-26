# -*- coding: utf-8 -*-
"""Untitled0.ipynb


Original file is located at
    https://colab.research.google.com/drive/1shk7VP6le3q4_V1CmJjlwlOvUNXH5Twa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df= pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')

df.head()

df.describe

df= df.drop(['Date', 'Holiday', 'Seasons'], axis=1)
df

df_cols= ['bike_count', 'hour', 'temp', 'humaidity', 'wind', 'visibility', 'dew_pt_temp', 'radiation', 'rain', 'snow', 'functional']
df.columns= df_cols
df

df['functional']= (df['functional']== 'Yes').astype(int)
df= df[df['hour']== 12]
df= df.drop(['hour'], axis=1)
df

for label in df.columns[1:]:
    plt.scatter(df[label], df['bike_count'])
    plt.title(label)
    plt.ylabel('Bike Count at Noon')
    plt.xlabel(label)
    plt.show()

df= df.drop(['wind', 'visibility', 'functional'], axis=1)

train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

#df = df.sample(frac=1).reset_index(drop=True)

#train_end = int(0.6 * len(df))
#valid_end = int(0.8 * len(df))

#train = df.iloc[:train_end]
#valid = df.iloc[train_end:valid_end]
#test = df.iloc[valid_end:]

def get_xy(dataframe, y_label, x_label= None):
    dataframe= copy.deepcopy(dataframe)
    if not x_label:
        x= dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_label)== 1:
            x= dataframe[x_label[0]].values.reshape(-1,1)
        else:
            x= dataframe[x_label].values

    y= dataframe[y_label].values.reshape(-1,1)
    data= np.hstack((x, y))

    return data, x, y

_, x_train_temp, y_train_temp= get_xy(train, 'bike_count', x_label=['temp'])
_, x_val_temp, y_val_temp= get_xy(val, 'bike_count', x_label=['temp'])
_, x_test_temp, y_test_temp= get_xy(test, 'bike_count', x_label=['temp'])

x_train_temp

temp_reg= LinearRegression()
temp_reg.fit(x_train_temp, y_train_temp)

print(temp_reg.coef_, temp_reg.intercept_)

temp_reg.score(x_test_temp, y_test_temp)

plt.scatter(x_train_temp, y_train_temp, label='Data', color='blue')
x= tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label= 'Fit', color='red', linewidth=3)
plt.legend()
plt.title('Bike vs Temp')
plt.ylabel('Number of Bikes')
plt.xlabel('Temp')
plt.grid(True)
plt.show()

"""### Multiple Linear Regrassion"""

def get_xy(dataframe, y_label, x_label= None):
    dataframe= copy.deepcopy(dataframe)
    if x_label is None:
        x= dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_label)== 1:
            x= dataframe[x_label[0]].values.reshape(-1,1)
        else:
            x= dataframe[x_label].values

    y= dataframe[y_label].values.reshape(-1,1)
    data= np.hstack((x, y))

    return data, x, y

df = df.sample(frac=1).reset_index(drop=True)

train_end = int(0.6 * len(df))
valid_end = int(0.8 * len(df))

train = df.iloc[:train_end]
valid = df.iloc[train_end:valid_end]
test = df.iloc[valid_end:]

_, x_train_all, y_train_all= get_xy(train, 'bike_count', x_label=df.columns[1:])
_, x_val_all, y_val_all= get_xy(val, 'bike_count', x_label=df.columns[1:])
_, x_test_all, y_test_all= get_xy(test, 'bike_count', x_label=df.columns[1:])

all_reg= LinearRegression()
all_reg.fit(x_test_all, y_test_all)

all_reg.score(x_test_all, y_test_all)

"""### Regression with Neural Network"""

def plot_loss(history):
    plt.plot(history.history['loss'], label= 'loss')
    plt.plot(history.history['val_loss'], label= 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

temp_normalizer= tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(x_train_temp.reshape(-1))

temp_nn_model= tf.keras.Sequential([
      temp_normalizer,
      tf.keras.layers.Dense(1)
])

temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

history= temp_nn_model.fit(
    x_train_temp.reshape(-1), y_train_temp,
    epochs= 100,
    validation_data= (x_val_temp, y_val_temp)
)

plot_loss(history)

history= temp_nn_model.fit(
    x_train_temp.reshape(-1), y_train_temp,
    epochs= 100,
    validation_data= (x_val_temp, y_val_temp)
)

plot_loss(history)

history= temp_nn_model.fit(
    x_train_temp.reshape(-1), y_train_temp,
    epochs= 1000,
    validation_data= (x_val_temp, y_val_temp)
)

plot_loss(history)

plt.scatter(x_train_temp, y_train_temp, label='Data', color='blue')
x= tf.linspace(-20, 40, 100)
plt.plot(x, temp_nn_model.predict(np.array(x).reshape(-1, 1)), label= 'Fit', color='red', linewidth=3)
plt.legend()
plt.title('Bike vs Temp')
plt.ylabel('Number of Bikes')
plt.xlabel('Temp')
plt.grid(True)
plt.show()

"""### Neural Nework"""

temp_normalizer= tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(x_train_temp.reshape(-1))

nn_model= tf.keras.Sequential([
      temp_normalizer,
      tf.keras.layers.Dense(32, activation= 'relu'),
      tf.keras.layers.Dense(32, activation= 'relu'),
      tf.keras.layers.Dense(1, activation= 'relu'),
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history= nn_model.fit(
    x_train_temp, y_train_temp,
    validation_data=(x_val_temp, y_val_temp),
    epochs=100
)

plot_loss(history)

plt.scatter(x_train_temp, y_train_temp, label='Data', color='blue')
x= tf.linspace(-20, 40, 100)
plt.plot(x, nn_model.predict(np.array(x).reshape(-1, 1)), label= 'Fit', color='red', linewidth=3)
plt.legend()
plt.title('Bike vs Temp')
plt.ylabel('Number of Bikes')
plt.xlabel('Temp')
plt.grid(True)
plt.show()

all_normalizer= tf.keras.layers.Normalization(input_shape=(6,), axis=None)
all_normalizer.adapt(x_train_all)

nn_model= tf.keras.Sequential([
      all_normalizer,
      tf.keras.layers.Dense(32, activation= 'relu'),
      tf.keras.layers.Dense(32, activation= 'relu'),
      tf.keras.layers.Dense(1, activation= 'relu'),
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history= nn_model.fit(
    x_train_all, y_train_all,
    validation_data=(x_val_all, y_val_all),
    epochs=100
)

plot_loss(history)

all_reg.predict(x_test_all)

y_pred_lr= all_reg.predict(x_test_all)
y_pred_nn= nn_model.predict(x_test_all)

def MSE(y_pred, y_real):
    return np.square(y_pred-y_real).mean()

MSE(y_pred_lr, y_test_all)

MSE(y_pred_nn, y_test_all)

plt.figure(figsize=(6, 6))
plt.scatter(y_test_all, y_pred_lr, label='Linear Reg. Pred', c='red')
plt.title('Linear Regression Prediction')
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims= [0, 1550]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_all, y_pred_nn, label='Neural Net Pred', c='blue')
plt.title('Neural Net. Prediction')
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims= [0, 1550]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_all, y_pred_lr, label='Linear Reg. Pred', c='red')
plt.scatter(y_test_all, y_pred_nn, label='Neural Net Pred', c='blue')
plt.legend()
plt.title('Bike vs Temp')
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims= [0, 1550]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
