import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from datetime import datetime

from dataLoader import dataLoader

class predictor:
    def __init__(self, trainPummokData, cfg):
        #@ calculate & save yearly trend for the correlated years
        #@ correlated years are wroten in cfg
        self.trends = []
        for idx in range(0, 36 + 1):
            splits = np.split(trainPummokData[idx]["해당일자_전체평균가격(원)"].to_numpy(), [365, 365*2, 365*3, 365*4])
            splits.pop(-1)

            ctitle = 'predictor_correlated'
            if not cfg[ctitle][idx]:
                self.trends.append(np.array([]))
                continue
            
            relIdx = [j-1 for j in cfg[ctitle][idx]]
            baseTrend = splits[relIdx[-1]]

            linreg = [LinearRegression().fit(np.reshape(splits[j], (-1,1)), baseTrend) for j in relIdx]
            avg = np.zeros(365)
            for regIdx, j in enumerate(relIdx):
                avg = avg + splits[j] * linreg[regIdx].coef_[0] + linreg[regIdx].intercept_
            avg = avg / len(relIdx)
            self.trends.append(avg)

    #@ returns prediction list
    #@ testPummokData: dataframe of a pummok csv, must include columns "datadate" and "해당일자_전체평균가격(원)"
    #@ trendtype: to which trend to use for prediction, no reason to be different from testPummokData's type
    #@ mode: "avg"
    def predictionFromTrend(self, testPummokData, trendtype, mode = "avg", pltReg = False):
        #@ return constant price if no trends or test data to predict from
        if self.trends[trendtype].size==0 or testPummokData.empty:
            return [1]*29
        
        dates = testPummokData["datadate"].tolist()
        prices = testPummokData["해당일자_전체평균가격(원)"].tolist()
        dateLocs = list(map(dateToDayLoc, dates))

        #@ extract prices that are not NaN
        isnan_ = np.isnan(prices)
        notNanDLocs = [dateLocs[idx] for idx, b in enumerate(isnan_) if not b]
        notNanPrices = [prices[idx] for idx, b in enumerate(isnan_) if not b]

        #@ get coefficients to be multiplied and added to trend data
        if notNanDLocs:
            if mode == "avg":
                alpha = 0
                beta = np.sum(notNanPrices) / np.sum([self.trends[trendtype][i] for i in notNanDLocs])

            if pltReg:
                print(f"@@@ plot for type {trendtype} @@@")
                print(f"test data date locations: {notNanDLocs}")
                print(f"test data prices: {notNanPrices}")
                print(f"alpha: {alpha}, beta: {beta}")

                x = np.array([self.trends[trendtype][i] for i in notNanDLocs])
                plt.scatter(x, notNanPrices)
                plt.scatter(alpha + beta * x, notNanPrices)
                plt.show()
        else:
            alpha = 0
            beta = 1

        base_dloc = dateLocs[-1]
        #@ the prediction, leap years shouldn't matter much so just do mod 365
        target_dates = ((np.arange(base_dloc, base_dloc+29))%365).tolist()
        typePriceAns = alpha + beta * np.array([self.trends[trendtype][i] for i in target_dates])
        #@ overwrite the price of 'day 0' if it was not empty
        if not np.isnan(prices[-1]):
                typePriceAns[0] = prices[-1]

        return typePriceAns.tolist()

#@ print correlation matrix for each type
def printCorrMatrix(trainPummokData):
    for idx in range(0, 36 + 1):
        splits = np.split(trainPummokData[idx]["해당일자_전체평균가격(원)"].to_numpy(), [365, 365*2, 365*3, 365*4])
        splits.pop(-1)
        print(f"\n{idx}")
        print(np.corrcoef([splits[0], splits[1], splits[2], splits[3]]))

        if 0:
            plt.plot(splits[0])
            plt.plot(splits[1])
            plt.plot(splits[2])
            plt.plot(splits[3])
            plt.legend(['1','2','3','4'])
            plt.title(idx)
            plt.show()

#@ maps date string in format "YYYYMMDD"
#@ to an integer in [0,364] ([0,365] for leap years)
def dateToDayLoc(dateStr):
    dS = dateStr.strip()
    date_format = "%Y%m%d"
    i = datetime.strptime(dS[:4]+"0101", date_format)
    f = datetime.strptime(dS, date_format)
    return (f - i).days

if __name__ == "__main__":
        #@ load config file
        config_module =	__import__('config')
        cfg= config_module.config
        
        #@ load raw train data
        trainData = dataLoader('./aT_train_raw/pummok_*.csv', saveFilePrefix = "train_", type = "csv")  #trainData = dataLoader('./euijun/data/prices/*.txt', type = "pptxt")
        trainData.load(save = False, load = True)
        #@ do euijun's preprocessing to train data
        trainData.ejsPreProcess()

        #@ load raw test data
        testDataSet = []
        for idx in range(0, 9 + 1):
            testDataSet.append(dataLoader(f'./aT_test_raw/sep_{idx}/pummok_*.csv', saveFilePrefix = f"test{idx}_", type = "csv"))
            testDataSet[idx].load(save = False, load = True)    #@ switch save & load arguments when in need

        #@ print yearly correlation matrix
        # printCorrMatrix(trainData.pummokData)

        #@ create predictor
        pred = predictor(trainData.pummokData, cfg)
        #@ predict
        allPriceAns = []
        for set in range(0, 9 + 1):
            setPriceAns = []
            for type in range(0, 36 + 1):
                setPriceAns.append(pred.predictionFromTrend(
                    testPummokData = testDataSet[set].pummokData[type],
                    trendtype = type,
                    mode = "avg"))
            allPriceAns.append(setPriceAns)