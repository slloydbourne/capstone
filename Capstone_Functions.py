#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import pickle

#Scikit-Learn Importing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer

#Import PyWavelets for WaveSmoothing
import pywt
import mad
from statsmodels import robust


class CapstoneFunctions:
    
    def __init__(self):
        pass
    
    @staticmethod

    def Train_Test_Split(Data1, Port, Window, Lookback, scaler): 

        """ 
        Import Monthly data with Date as Index. Select which portfolio to train test split. Set number of
        lookback months and whether to use MinMaxScaler.

        Example:
        HiXdata, HiYdata, HiXtrain, HiYtrain, HiXtest, HiYtest, HiPredictedX, HiForecastX, HiX, HiY = \
        Train_Test_Split(Monthly, 'Hi', 60, 12, scaler = False)

        """ 


        Data1 = Data1.values

        TrainSplit = int(Window*0.90)
        TestSplit = int(Window*0.1)

        #Iterate Through Data and Creat Numpy arrays with 12 Months of \ 
        #lagged data for X and 13th Month for Y
        tmpX=[]
        tmpY=[]
        for A in range(len(Data1)-Lookback):
            tmp=Data1[A:(A + Lookback),Port]
            tmpX.append(tmp)
            tmpY.append(Data1[(A + Lookback),Port])
        Xdata = np.array(tmpX)
        Ydata = np.array(tmpY)

        tmpForeX=[]
        for A in range(len(Data1)-Lookback):
            tmpFore=Data1[A+1:(A+1 + Lookback),Port]
            tmpForeX.append(tmpFore)
        XdataFore = np.array(tmpForeX)

        Ydata = Ydata.reshape(len(Ydata),1)

        Shape = int(Xdata.shape[0])
        Shape2 = int(XdataFore.shape[0])


        #Create Empty Numpy Arrays
        Xtrain=np.ones(shape=(Shape-Window,TrainSplit,12))
        Ytrain=np.ones(shape=(Shape-Window,TrainSplit,1))
        Xtest=np.ones(shape=(Shape-Window,TestSplit,Lookback))
        Ytest=np.ones(shape=(Shape-Window,TestSplit,1))
        PredictedX=np.ones(shape=(Shape-Window,1,Lookback))
        ForecastX=np.ones(shape=(Shape2-Window,1,Lookback))
        X = np.ones(shape=(Shape-Window,Window,Lookback))
        Y = np.ones(shape=(Shape-Window,Window,1))

        #Fill Numpy Arrays with data
        for B in range(Shape-Window):
            Xtrain[B,:,:] = Xdata[B:B+TrainSplit,:]
            Ytrain[B,:,:] = Ydata[B:B+TrainSplit]
            Xtest[B,:,:] = Xdata[B+TrainSplit:B+Window,:]
            Ytest[B,:,:] = Ydata[B+TrainSplit:B+Window]
            PredictedX[B,:,:] = Xdata[B+Window,:]
            ForecastX[B,:,:] = XdataFore[B+Window,:]
            X[B,:,:] = Xdata[B:B+Window,:]
            Y[B,:,:] = Ydata[B:B+Window]

            if scaler:
                sc = MinMaxScaler()
                Xtrain[B,:,:] = sc.fit_transform(Xtrain[B,:,:])
                Xtest[B,:,:] = sc.transform(Xtest[B,:,:])
                PredictedX[B,:,:] = sc.transform(PredictedX[B,:,:])
                X[B,:,:] = sc.transform(X[B,:,:])



        return Xdata, Ydata, Xtrain, Ytrain, Xtest, Ytest, PredictedX, ForecastX, X, Y


    @staticmethod


    def WaveSmooth(Data):

        """ 
        Use Wave Smoothing algorithms to smooth Monthly data before using Train/Test Split and using the LSTM algo

        example WaveSmooth(Monthly)

        """ 

        Data2 = Data.copy()
        Data3 = Data.copy()
        Data3 = Data3.reset_index()
        Datahi = Data2.iloc[:,0].copy()
        Datalo = Data2.iloc[:,1].copy()

        coeffhi = pywt.wavedec(Datahi, "haar", mode="per", level=2)
        Sigma = robust.mad(coeffhi[-2])
        Threshold = Sigma * np.sqrt(2*np.log(len(Datahi)))
        coeffhi[1:] = (pywt.threshold(i, value=Threshold, mode="soft") for i in coeffhi[1:])
        hi = pywt.waverec(coeffhi, "haar", mode="per" )

        coefflo = pywt.wavedec(Datalo, "haar", mode="per", level=2)
        Sigma = robust.mad(coefflo[-2])
        thresh = Sigma * np.sqrt(2*np.log(len(Datalo)))
        coefflo[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coefflo[1:])
        lo = pywt.waverec(coefflo, "haar", mode="per" )

        WaveData = pd.DataFrame(hi)
        WaveData['tmpHi'] = pd.DataFrame(hi)
        WaveData['tmpLo'] = pd.DataFrame(lo)
        Data3['hi'] = WaveData['tmpHi'].copy()
        Data3['lo'] = WaveData['tmpLo'].copy()
        Data3.set_index('date', inplace=True)

        return Data3


    @staticmethod


    def VolatilityScale (Daily, Monthly):
        """ 
        Takes Daily data and sums up by month (21 business days). This is squared and 6 month intervals are summed. 
        Finally this is multiplied by 21/126. This Volatility is square root to Standard deviation. 
        Each entry of the final dataframe should be multiplied by output of algos for volatility scaling.

        example VolatilityScale (Daily, Monthly)

        """    


        DailyData = Daily.copy()
        MonthlyData = Monthly.copy()

        DailyData2 = DailyData.resample('BMS').sum()**2
        def summing(X):
            X = X.sum()
            return X
        DailyData = ((DailyData2.rolling(6).apply(summing))*(21/126))**(1/2)
        DailyData = (0.12/DailyData)
        MonthlyData = MonthlyData.multiply(DailyData)
        return MonthlyData

