import pandas as pd
import numpy as np
import tensorflow as tf

#This is the prediction, it follows the simple y = xm + b format, where x is the year
def prediction(b,m,year):
    predicted_price = (b + (m * year))
    return predicted_price

#The cost algorithm I used is the “Mean squared error”:
def cost(b, m, years, prices):
    length = float(len(years))
    endCost = 0
    summation = 0

    for i in range (0,len(prices)):
        difference = prices[i] - prediction(b, m, float(years[i]))
        squared_difference = difference**2 
        summation = summation + squared_difference
    endCost = summation/len(prices)

    return endCost

#This is gradient descent using the batch method. Here we find the derivates of b and m.
def batch_gradient_descent(b, m, years, prices, alpha, epochs):
    bTraining = b
    mTraining = m
    bMid = 0
    mMid = 0
    numEntries = len(years)

    for convergence in range(epochs):

        for i in range(numEntries):
            bMid += (prediction(bTraining, mTraining, float(years[i])) - float(prices[i])) * (1/float(numEntries))
            mMid += (prediction(bTraining, mTraining, float(years[i])) - float(prices[i])) * (1/float(numEntries)) * float(years[i])

        mTraining = (m - (alpha * mMid))
        bTraining = (b - (alpha * bMid))

    return bTraining, mTraining

#main function that will read the data, run the gradient descents, then ask you for a year to predict the price of gold in
def main():
    workbook = pd.read_csv('annual_csv.csv', usecols = ['Price','Date'])
    workbook.head()
    dateList = workbook['Date'].tolist()
    yearList = range(len(dateList))
    priceList = workbook['Price'].tolist()

    b = 0
    m = 0
    points = []
    epochs = 1000
    alpha = 0.001

    bTrained, mTrained = batch_gradient_descent(b, m, yearList, priceList, alpha, epochs)

    endCost = cost(bTrained, mTrained, yearList, priceList)

    print("Weight of b: " + str(bTrained))
    print("Weight of m: " + str(mTrained))
    print("Loss: " + str(endCost))

    # uncomment to get predicted values
    #
    # pred = ""
    # guess = 0
    # while (pred != 'end'):
    #     pred = input("Enter a year to predict the price of gold for : ")
    #     if (pred.strip().isdigit()):
    #         guess = prediction(bTrained, mTrained, float(pred)-1950)
    #         print("The predicted price is: " + str(guess))
    #     else:
    #         print("Please enter a year, or end to stop.")


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

if __name__ == '__main__':
    main()
    
