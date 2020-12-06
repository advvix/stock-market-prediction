import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler

# fix random seed for reproducibility
np.random.seed(7)

# Import data
data = pd.read_csv('last.csv')
# Drop date variable
#data = data.drop(['DATE'], 1)

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
print(p)
# Make data a numpy array
data = data.values

# normalize the dataset
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# split into train and test sets
train_start = 0
train_end = int(np.floor(0.9 * n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# reshape input to be [samples, time steps, features]
trainX = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape[0],X_train.shape[1])
# create and fit the LSTM network
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(16,  return_sequences=True))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(1, activation='linear'))
opt = tf.keras.optimizers.Adam(learning_rate=0.00315)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mape'])
history = model.fit(trainX, y_train, epochs=200, validation_split=0.05, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#print('Train score:',np.sqrt(history.history['loss']))
#print('Test score:',np.sqrt(history.history['val_loss']))

plt.title('Mean Squared Error')
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#plt.title('WIG20')
plt.plot(testPredict,label='predict')
plt.plot(y_test,label='true')
plt.legend()
plt.show()
