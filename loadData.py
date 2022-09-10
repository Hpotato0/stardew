# THIS CODE WILL SAVE/LOAD PREPROCESSED DATA FROM ./pp_data/pummok

from glob import glob
from tqdm import tqdm
from pandasql import sqldf
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utility_submit import *



class pummokLoader(object):
    #@ dir: aT_train_raw/pummok_*.csv
    def __init__(self, dir, saveFilePrefix = "", type = "raw"):
        self.data_list = glob(dir) 
        self.pummok_list = glob(dir) #@ list of pummok csv directories, in random order  
        self.pummokData = [None]*(37) #@ placeholder for each type's data, in correct order
        self.saveFilePrefix = saveFilePrefix
        self.type = type #@ what type of data to load

    def load(self, save = False, load = False):
        if self.type == "raw":
            self.loadFromRawCsv(save = save, load = load)
        elif self.type == "pptxt":
            self.loadFromPPrTxt()


    #@ load pummok data from csv file
    #@ save/load simplified data in ./{dirName}/pummok, dirName = "pp_data" by default
    def loadFromRawCsv(self, save = False, load = False):
        dirName = "pp_data"

        #@ iterate through csvs for different types
        for filepath in tqdm(self.pummok_list):
            name = filepath.split('/')[-1] #@ e.g ".../pummok_5.csv" -> "pummok_5.csv"
            savePath = f"./{dirName}/pummok/{self.saveFilePrefix}{name}"

            #@ load
            if load:
                ddf = pd.read_csv(savePath, dtype = {"datadate": str, "해당일자_전체평균가격(원)": float})
                self.pummokData[int(name.split("_")[1].split(".")[0])] = ddf.copy()
                continue

            #@ extract columns of interest + some very basic preprocessing
            ddf = pd.read_csv(filepath, usecols = ["datadate", "해당일자_전체평균가격(원)"], dtype = {"datadate": str, "해당일자_전체평균가격(원)": float})
            ddf = sqldf(f"select * from ddf group by datadate")
            ddf = ddf.fillna(value=np.nan) #@ replace all 'None' with 'NaN'

            #@ put extracted data into class variable
            self.pummokData[int(name.split("_")[1].split(".")[0])] = ddf.copy()

            #@ save
            if save:
                if os.path.exists(f'./{dirName}') == False:
                    os.mkdir(f'./{dirName}')
                if os.path.exists(f'./{dirName}/pummok') == False:
                    os.mkdir(f'./{dirName}/pummok')

                ddf.to_csv(savePath, index=False)

    #@ load txt data created by joykang's preprocessor
    def loadFromPPrTxt(self):
        for filepath in tqdm(self.pummok_list):
            name = filepath.split('/')[-1]
            with open(filepath, "r") as pricefile:
                price = np.loadtxt(pricefile)
            self.pummokData[int(name.split(".")[0])] = price


#@ maps date string in format "YYYYMMDD"
#@ to an integer in [0,364] ([0,365] for leap years)
def dateToDayLoc(dateStr):
    dS = dateStr.strip()
    date_format = "%Y%m%d"
    i = datetime.strptime(dS[:4]+"0101", date_format)
    f = datetime.strptime(dS, date_format)
    return (f - i).days

if __name__ == "__main__":
    #@ load euijun's preprocesse data
    trainData = pummokLoader('./euijun/data/prices/*.txt', type = "pptxt")
    trainData.load()

    #@ load raw test data
    testDataSet = []
    for idx in range(0, 9 + 1):
        testDataSet.append(pummokLoader(f'./aT_test_raw/sep_{idx}/pummok_*.csv', saveFilePrefix = f"test{idx}_", type = "raw"))
        testDataSet[idx].load(save = False, load = True)    #@ switch save & load arguments when in need


    config_module =	__import__('config')
    cfg= config_module.config
    ctitle = 'correlated'

    
    trends = []
    for idx in range(0, 36 + 1):
        splits = np.split(trainData.pummokData[idx], [365, 365*2, 365*3, 365*4])
        splits.pop(-1)

        #@ print correlation matrix for each type
        #print(f"\n{idx}")
        #print(np.corrcoef([splits[0], splits[1], splits[2], splits[3]]))

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
            #     lr_predictor = LinearRegression().fit(np.reshape([trends[type][i] for i in notNanDLocs], (-1,1)), notNanPrices)
            #     alpha = lr_predictor.intercept_
            #     beta = lr_predictor.coef_[0]
            #     print(alpha)
            #     print(beta)

            if notNanDLocs:
                lr_predictor = LinearRegression().fit(np.reshape([trends[type][i] for i in notNanDLocs], (-1,1)), notNanPrices)
                alpha = lr_predictor.intercept_
                beta = lr_predictor.coef_[0]
            else:
                alpha = 0
                beta = 1

            alpha = 0
            beta = np.sum(notNanPrices) / np.sum([trends[type][i] for i in notNanDLocs])

            base_dloc = dateLocs[-1]

            #@ leap years shouldn't matter much
            target_dates = ((np.arange(base_dloc, base_dloc+29))%365).tolist()
            typePriceAns = alpha + beta * np.array([trends[type][i] for i in target_dates])
            
            if not np.isnan(prices[-1]):
                typePriceAns[0] = prices[-1]

            setPriceAns.append(typePriceAns.tolist())
        allPriceAns.append(setPriceAns)
    

    pricelistToCsv(allPriceAns, "zzb_yearly0")