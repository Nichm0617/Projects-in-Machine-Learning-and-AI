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

#cost algorithm used for stocashtic. Makes use of dot.
def  cost_function_theta(theta,y,x):
    hypothesese = x.dot(theta)
    summing = np.sum(np.square(hypothesese - y))
    total = (1/2 * len(y)) * summing
    return total


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


#The stocashtic version of gradient descent
def stocashtic_gradient_descent(theta, x, y, iters, learningRate):
    allCosts = np.zeros(iters)

    for iter in range(iters):
        cost = 0.0
        for i in range(len(y)):
            random_index = np.random.randint(0, len(y))
            yMid = y[random_index,0].reshape(1,1)
            xMid = x[random_index,0].reshape(1,x.shape[1])
            hypothesis = np.dot(xMid,theta)

            theta -= (1/len(y)) * (xMid.T.dot((hypothesis - yMid))) * learningRate
            cost = cost + (cost_function_theta(theta,yMid,xMid))
        allCosts[iter] = cost
    finalTheta = theta

    return allCosts, finalTheta

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

    print("Batch Gradient Descent:")
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


    #populate arrays
    theta = np.random.randn(1,2)
    x = np.random.rand(70,1)
    y = np.random.randn(70,1)
    for num in range(len(yearList)):
        x[num][0] = yearList[num]
        y[num][0] = priceList[num]

    allCosts, finalTheta = stocashtic_gradient_descent(theta, x, y, 50, 0.001)

    print("Stocashtic Gradient Descent:")
    print("Weight of b: " + str(finalTheta[0][0]))
    print("Weight of m: " + str(finalTheta[0][1]))
    print("Loss: " + str(allCosts[-1]))


if __name__ == '__main__':
    main()
