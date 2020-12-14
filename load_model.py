import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Wczytanie danych oraz zapisanie wielkości tablicy do zmiennych
data = pd.read_csv('data.csv')
n = data.shape[0]
p = data.shape[1]
data = data.values

# Normalizacja
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Podział na dane treningowe i testowe
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

# wczytanie modelu
model = tf.keras.models.load_model('my_model.json')

# predykcja
testPredict = model.predict(testX)

testPred = []
for i in range(len(y_test)):
    testPred.append(testPredict[:][i][0].astype(np.float))

plt.title('AAP')
plt.plot(testPredict,label='predict')
plt.plot(y_test,label='true')
plt.legend()
plt.show()

loss = tf.keras.losses.mean_squared_error(y_test, testPred).numpy()
print('Błąd średniokwadratowy:', loss)
