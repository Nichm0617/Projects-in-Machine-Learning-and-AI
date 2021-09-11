import pandas as pd
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    workbook = pd.read_csv('annual_csv.csv', usecols = ['Price','Date'])
    workbook.head()
    dateList = workbook['Date'].tolist()
    yearList = range(len(dateList))
    priceList = workbook['Price'].tolist()

    print("Adam:")
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    m.compile(opt, loss='mse')
    data = np.arange(2*len(yearList)).reshape(2, len(yearList))
    for num in range(len(yearList)):
        data[1][num] = yearList[num]
        data[0][num] = priceList[num]
    labels = np.zeros(2)
    results = m.fit(data, labels)


    print("RMSprop:")
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    m.compile(opt, loss='mse')
    data = np.arange(2*len(yearList)).reshape(2, len(yearList))
    for num in range(len(yearList)):
        data[1][num] = yearList[num]
        data[0][num] = priceList[num]
    labels = np.zeros(2)
    results = m.fit(data, labels)