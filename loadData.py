# THIS CODE WILL SAVE/LOAD PREPROCESSED DATA FROM ./pp_data/pummok

from glob import glob
from tqdm import tqdm
from pandasql import sqldf
from sklearn.linear_model import LinearRegression, Ridge
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

from utility_submit import *


class dataLoader(object):
    #@ dir: aT_train_raw/pummok_*.csv
    def __init__(self, dir, saveFilePrefix = "", type = "csv"):
        self.data_list = glob(dir) 
        self.pummok_list = glob(dir) #@ list of pummok csv directories, in random order  
        self.pummokData = [None]*(37) #@ placeholder for each type's data, in correct order
        self.saveFilePrefix = saveFilePrefix
        self.type = type #@ what type of data to load

    def load(self, save = False, load = False):
        if self.type == "csv":
            self.loadFromCsv(save = save, load = load)
        elif self.type == "pptxt":
            self.loadFromPPrTxt()


    #@ load pummok data from csv file
    #@ save/load simplified data in ./{dirName}/pummok, dirName = "pp_data" by default
    def loadFromCsv(self, save = False, load = False):
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

    #@ fill single NaNs with the average of date before & after
    #@ fill multiple NaNs with linear interpolation
    def fillNaN(self):
        for tidx, ddf in enumerate(self.pummokData):
            price = ddf["해당일자_전체평균가격(원)"].to_numpy()

            #@ remove outliers using median filter
            nz = np.reshape(np.argwhere(~np.isnan(price)), (1,-1))[0]
            price[nz] = ndimage.median_filter(price[nz], size = 5)
            
            #@ fill single NaNs with the average of data before & after
            if np.isnan(price[0]) and not np.isnan(price[1]):
                price[0] = price[1]
            for i in range(1, len(price)-1):
                if np.isnan(price[i]) and not np.isnan(price[i-1]) and not np.isnan(price[i+1]):
                    price[i] = (price[i-1] + price[i+1])/2
            if np.isnan(price[-1]) and not np.isnan(price[-2]):
                price[-1] = price[-2]
            
            #@ fill multiple NaNs with linear interpolation
            if 0:
                start = 0
                end = 0
                interp = False
                for idx, p in enumerate(price):
                    if not interp and np.isnan(p):
                        start = idx
                        interp = True
                    elif interp and (not np.isnan(p) or idx==len(price)-1):
                        end = idx
                        interp = False

                        if start == 0: #@ in case NaN starts at front
                            p_s = p
                        else:
                            p_s = price[start-1]

                        if end==len(price)-1 and np.isnan(p): #@ in case NaN ends at end
                            p_e = price[start-1]
                        else:
                            p_e = p

                        a = np.array([p_s + (p_e-p_s)/(end-start+1)*x for x in range(1,end-start+2)])
                        price[start:end+1] = np.array([p_s + (p_e-p_s)/(end-start+1)*x for x in range(1,end-start+2)])
            else:
                nz = np.reshape(np.argwhere(~np.isnan(price)), (1,-1))[0]
                price = [x if not np.isnan(x) else price[nz[np.argmin(np.abs(i-nz))]] for i, x in enumerate(price)]

            ddf.loc[:,"해당일자_전체평균가격(원)"] = price



#@ maps date string in format "YYYYMMDD"
#@ to an integer in [0,364] ([0,365] for leap years)
def dateToDayLoc(dateStr):
    dS = dateStr.strip()
    date_format = "%Y%m%d"
    i = datetime.strptime(dS[:4]+"0101", date_format)
    f = datetime.strptime(dS, date_format)
    return (f - i).days

if __name__ == "__main__":
    #@ load euijun's preprocessed data
    trainData = dataLoader('./euijun/data/prices/*.txt', type = "pptxt")
    trainData.load()

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