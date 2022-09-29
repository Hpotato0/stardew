import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from constrained_linear_regression import ConstrainedLinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage

from dataLoader import dataLoader
from utility_submit import pricelistToCsv

class predictor:
    def __init__(self, trainPummokData, cfg):
        #@ calculate & save yearly trend for the correlated years
        #@ correlated years are wroten in cfg
        ctitle = "predictor_correlated"
        self.data_type = cfg["predictor_correlated"]["data_type"]

        self.trends = []
        for idx in range(0, len(trainPummokData)):
            if not cfg[ctitle][idx]:
                self.trends.append(np.array([]))
                continue

            if self.data_type == "mvavg":
                splits = np.split(trainPummokData[idx], [365, 365*2, 365*3, 365*4])
            elif self.data_type == "raw":
                splits = np.split(trainPummokData[idx]["해당일자_전체평균가격(원)"].to_numpy(), [365, 365*2, 365*3, 365*4])
            
            splits.pop(-1)

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
        prices_nofilter = prices
        if self.data_type == "mvavg":  
            prices = ndimage.median_filter(prices, size=3)
            prices = ndimage.median_filter(prices, size=5)
        dateLocs = list(map(dateToDayLoc, dates))

        #@ extract prices that are not NaN
        isnan_ = np.isnan(prices)
        notNanDLocs = [dateLocs[idx] for idx, b in enumerate(isnan_) if not b]
        notNanPrices = [prices[idx] for idx, b in enumerate(isnan_) if not b]

        prices_smooth = prices
        
        #     notNanPrices = [prices_smooth[idx] for idx, b in enumerate(isnan_) if not b]

        #@ get coefficients to be multiplied and added to trend data
        if notNanDLocs:
            if mode == "avg":
                alpha = 0
                beta = np.sum(notNanPrices) / np.sum([self.trends[trendtype][i] for i in notNanDLocs])

            elif mode == "ejchung": #@ sometimes doesn't converge
                regr = ConstrainedLinearRegression()
                regr.fit(np.reshape([self.trends[trendtype][i] for i in notNanDLocs], (-1,1)), notNanPrices,
                max_coef = [1.2],
                min_coef = [0.4])
                alpha = regr.intercept_
                beta = regr.coef_[0]

            elif mode == "naive":
                lr_predictor = LinearRegression().fit(np.reshape([self.trends[trendtype][i] for i in notNanDLocs], (-1,1)), notNanPrices)
                #lr_predictor = Ridge(alpha=10000000).fit(np.reshape([trends[type][i] for i in notNanDLocs], (-1,1)), notNanPrices)
                alpha = lr_predictor.intercept_
                beta = lr_predictor.coef_[0]
                

            #@ plot the regression
            # if pltReg:
            #     print(f"@@@ plot for type {trendtype} @@@")
            #     print(f"test data date locations: {notNanDLocs}")
            #     print(f"test data prices: {notNanPrices}")
            #     print(f"alpha: {alpha}, beta: {beta}")

            #     x = np.reshape([self.trends[trendtype][i] for i in notNanDLocs], (-1,1))
            #     plt.scatter(x, notNanPrices)
            #     plt.scatter(x, alpha+beta*x)
            #     plt.show()
        else:
            alpha = 0
            beta = 1

        base_dloc = dateLocs[-1]
        #@ the prediction, leap years shouldn't matter much so just do mod 365
        target_dates = ((np.arange(base_dloc, base_dloc+29))%365).tolist()
        typePriceAns = alpha + beta * np.array([self.trends[trendtype][i] for i in target_dates])

        if self.data_type == "mvavg":
            if (typePriceAns[1] - prices_smooth[-1]) * (typePriceAns[1] - self.trends[trendtype][target_dates[1]]) < 0:
                c = 0.4
            else:
                c = 0.1
            typePriceAns = typePriceAns - (typePriceAns[1] - prices_nofilter[-1]) * c


        if pltReg:
            plt.figure(figsize = (30, 18))
            plt.plot(np.arange(base_dloc+1, base_dloc+29), typePriceAns[1:])
            x_trend = notNanDLocs + [base_dloc+i for i in range(1,30)]
            plt.plot(x_trend, [self.trends[trendtype][i%365] for i in x_trend])
            plt.plot(notNanDLocs, notNanPrices)
            plt.legend(['prediction', 'trend', 'price'])
            plt.title(f"{trendtype}")
            plt.show()


        #@ overwrite the price of 'day 0' if it was not empty
        if not np.isnan(prices_nofilter[-1]):
            typePriceAns[0] = prices_nofilter[-1]

        return typePriceAns.tolist()

#@ print correlation matrix for each type
def printCorrMatrix(trainPummokData):
    for idx in range(0, len(trainPummokData)):
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
        trainData.ejsPreprocess(medianFilterSize = 5)
        #@ calculate moving average
        trainData.createMovingAvg(30)

        #@ load raw test data
        testDataSet = []
        for idx in range(0, 9 + 1):
            testDataSet.append(dataLoader(f'./aT_test_raw/sep_{idx}/pummok_*.csv', saveFilePrefix = f"test{idx}_", type = "csv"))
            testDataSet[idx].load(save = False, load = True)    #@ switch save & load arguments when in need

        #@ print yearly correlation matrix
        # printCorrMatrix(trainData.pummokData)

        #@ create predictor
        data_type = cfg["predictor_correlated"]["data_type"]
        if data_type == "raw":
            pred = predictor(trainData.pummokData, cfg)
        elif data_type == "mvavg":
            pred = predictor(trainData.mvavg, cfg)
        
        #@ predict and save the results to allPriceAns
        allPriceAns = []
        for set in range(0, 9 + 1):
            setPriceAns = []
            for type in range(0, 36 + 1):
                # print(f"{set},{type}")
                mode = "avg" if set==7 and type==24 else "ejchung"
                setPriceAns.append(pred.predictionFromTrend(
                    testPummokData = testDataSet[set].pummokData[type],
                    trendtype = type,
                    mode = mode,
                    pltReg = (set==6)))
            allPriceAns.append(setPriceAns)

        #@ put allPriceAns into csv file in answer format
        pricelistToCsv(allPriceAns, csvName = "zzb_tmp")