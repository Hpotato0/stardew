from glob import glob
from tqdm import tqdm
from pandasql import sqldf
from sklearn.linear_model import LinearRegression, Ridge
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utility_submit import *
from loadData import *

if __name__ == "__main__":
    #@ load raw train data
    trainData = dataLoader('./aT_train_raw/pummok_*.csv', saveFilePrefix = "train_", type = "csv")
    trainData.load(save = False, load = True)

    #@ do euijun's preprocessing
    trainData.fillNaN()

    #@ load raw test data
    testDataSet = []
    for idx in range(0, 9 + 1):
        testDataSet.append(dataLoader(f'./aT_test_raw/sep_{idx}/pummok_*.csv', saveFilePrefix = f"test{idx}_", type = "csv"))
        testDataSet[idx].load(save = False, load = True)    #@ switch save & load arguments when in need

    #@ load config file
    config_module =	__import__('config')
    cfg= config_module.config
    ctitle = 'correlated'

    #@ average the correlated yearly data (those in config.py )
    trends = []
    for idx in range(0, 36 + 1):
        splits = np.split(trainData.pummokData[idx]["해당일자_전체평균가격(원)"].to_numpy(), [365, 365*2, 365*3, 365*4])
        splits.pop(-1)

        #@ print correlation matrix for each type
        print(f"\n{idx}")
        print(np.corrcoef([splits[0], splits[1], splits[2], splits[3]]))

        plt.plot(splits[0])
        plt.plot(splits[1])
        plt.plot(splits[2])
        plt.plot(splits[3])
        plt.legend(['1','2','3','4'])
        plt.title(idx)
        plt.show()

        if not cfg[ctitle][idx]:
            trends.append(np.array([]))
            continue
        
        relIdx = [j-1 for j in cfg[ctitle][idx]]
        baseTrend = splits[relIdx[-1]]

        linreg = [LinearRegression().fit(np.reshape(splits[j], (-1,1)), baseTrend) for j in relIdx]
        avg = np.zeros(365)
        for regIdx, j in enumerate(relIdx):
            avg = avg + splits[j] * linreg[regIdx].coef_[0] + linreg[regIdx].intercept_
        avg = avg / len(relIdx)
        trends.append(avg)

    #@ predict 
    allPriceAns = []
    for set in range(0, 9 + 1):
        setPriceAns = []
        for type in range(0, 36 + 1):
            if trends[type].size==0 or testDataSet[set].pummokData[type].empty:
                setPriceAns.append(([1]*29))
                continue

            dates = testDataSet[set].pummokData[type]["datadate"].tolist()
            prices = testDataSet[set].pummokData[type]["해당일자_전체평균가격(원)"].tolist()
            dateLocs = list(map(dateToDayLoc, dates))

            isnan_ = np.isnan(prices)
            notNanDLocs = [dateLocs[idx] for idx, b in enumerate(isnan_) if not b]
            notNanPrices = [prices[idx] for idx, b in enumerate(isnan_) if not b]
            

            # if type == 0 and set == 0:
            #     print(notNanDLocs)
            #     print(notNanPrices)
            #     x = [trends[type][i] for i in notNanDLocs]
            #     plt.scatter(x, notNanPrices)
            #     plt.show()
            #     lr_predictor = Ridge(alpha=1000000).fit(np.reshape([trends[type][i] for i in notNanDLocs], (-1,1)), notNanPrices)
            #     alpha = lr_predictor.intercept_
            #     beta = lr_predictor.coef_[0]
            #     print(alpha)
            #     print(lr_predictor.coef_)

            if notNanDLocs:
                # lr_predictor = LinearRegression().fit(np.reshape([trends[type][i] for i in notNanDLocs], (-1,1)), notNanPrices)
                # #lr_predictor = Ridge(alpha=10000000).fit(np.reshape([trends[type][i] for i in notNanDLocs], (-1,1)), notNanPrices)
                # alpha = lr_predictor.intercept_
                # beta = lr_predictor.coef_[0]

                alpha = 0
                beta = np.sum(notNanPrices) / np.sum([trends[type][i] for i in notNanDLocs])
            else:
                alpha = 0
                beta = 1

            

            base_dloc = dateLocs[-1]

            #@ leap years shouldn't matter much
            target_dates = ((np.arange(base_dloc, base_dloc+29))%365).tolist()
            typePriceAns = alpha + beta * np.array([trends[type][i] for i in target_dates])
            
            if not np.isnan(prices[-1]):
                typePriceAns[0] = prices[-1]

            setPriceAns.append(typePriceAns.tolist())
        allPriceAns.append(setPriceAns)
    

    pricelistToCsv(allPriceAns, "zzb_yearly0")