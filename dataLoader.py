# THIS CODE WILL SAVE/LOAD PREPROCESSED DATA FROM ./pp_data/pummok

from glob import glob
from tqdm import tqdm
from pandasql import sqldf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

from utility_submit import *


class dataLoader:
    #@ dir: aT_train_raw/pummok_*.csv
    def __init__(self, dir, saveFilePrefix = "", type = "csv"):
        self.pummok_list = glob(dir) #@ list of pummok csv directories, in random order  
        self.pummokData = [None]*(len(self.pummok_list)) #@ placeholder for each type's data, in correct order
        self.saveFilePrefix = saveFilePrefix
        self.type = type #@ what type of data to load

    #@ unified load function
    #@ data is loaded into self.pummokData
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

    #@ load txt data created by ejchung's preprocessor
    def loadFromPPrTxt(self):
        for filepath in tqdm(self.pummok_list):
            name = filepath.split('/')[-1]
            with open(filepath, "r") as pricefile:
                price = np.loadtxt(pricefile)
            self.pummokData[int(name.split(".")[0])] = price

    #@ ejchung's preprocessing code migrated to class method
    #@ i.e. fill NaNs & remove outliers via a median filter
    #@ the preprocessing is done on self.pummokData
    def ejsPreprocess(self, medianFilterSize = 5):
        for tidx, ddf in enumerate(self.pummokData):
            price = ddf["해당일자_전체평균가격(원)"].to_numpy()

            #@ manual removal of outliers
            if tidx==0:
                ddf.loc[ddf["datadate"]=="20130102","해당일자_전체평균가격(원)"] = np.nan
            if tidx == 1:
                ddf.loc[ddf["datadate"]=="20140211","해당일자_전체평균가격(원)"] = np.nan
                ddf.loc[ddf["datadate"]=="20150901","해당일자_전체평균가격(원)"] = np.nan

            #@ remove outliers using median filter
            nz = np.reshape(np.argwhere(~np.isnan(price)), (1,-1))[0]
            if tidx not in [13,3,1]:#for "raw"[7,13,21,3,8,15,26,36]:
                price[nz] = ndimage.median_filter(price[nz], size = medianFilterSize)
            
            #@ fill single NaNs with the average of data before & after
            if np.isnan(price[0]) and not np.isnan(price[1]):
                price[0] = price[1]
            for i in range(1, len(price)-1):
                if np.isnan(price[i]) and not np.isnan(price[i-1]) and not np.isnan(price[i+1]):
                    price[i] = (price[i-1] + price[i+1])/2
            if np.isnan(price[-1]) and not np.isnan(price[-2]):
                price[-1] = price[-2]
             
            if 1:
                #@ fill multiple NaNs with the nearest non-NaN value
                nz = np.reshape(np.argwhere(~np.isnan(price)), (1,-1))[0]
                price = [x if not np.isnan(x) else price[nz[np.argmin(np.abs(i-nz))]] for i, x in enumerate(price)]
            else:
                #@ fill multiple NaNs with linear interpolation
                #@ doesn't work as well..?
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
  
            ddf.loc[:,"해당일자_전체평균가격(원)"] = price

    #@ save pummokData (in case it was changed in ejsPreProcess)
    def savePPData(self, dirName = "pp_data2"):
        for filepath, ddf in zip(self.pummok_list, self.pummokData):
            name = filepath.split('/')[-1] #@ e.g ".../pummok_5.csv" -> "pummok_5.csv"
            savePath = f"./{dirName}/pummok/{self.saveFilePrefix}{name}"

            if os.path.exists(f'./{dirName}') == False:
                os.mkdir(f'./{dirName}')
            if os.path.exists(f'./{dirName}/pummok') == False:
                os.mkdir(f'./{dirName}/pummok')

            ddf.to_csv(savePath, index=False)

    #@ calculate and save moving average
    #@ must be used after ejsPreprocess or some other way of removing NaNs
    def createMovingAvg(self, size=10):
        self.mvavg = []
        for ddf in self.pummokData:
            price = ddf["해당일자_전체평균가격(원)"].to_numpy()
            price = ndimage.uniform_filter(price, size)
            self.mvavg.append(price)
        return

if __name__ == "__main__":
    #@ -------------------------
    #@ dataLoader example codes
    #@ -------------------------

    #@ load raw train data
    trainDataRaw = dataLoader('./aT_train_raw/pummok_*.csv', saveFilePrefix = "train_", type = "csv")  #trainData = dataLoader('./euijun/data/prices/*.txt', type = "pptxt")
    trainDataRaw.load(save = False, load = True)

    #@ load raw train data and do euijun's preprocessing
    trainDataPP = dataLoader('./aT_train_raw/pummok_*.csv', saveFilePrefix = "train_", type = "csv")  #trainData = dataLoader('./euijun/data/prices/*.txt', type = "pptxt")
    trainDataPP.load(save = False, load = True)
    trainDataPP.ejsPreprocess(medianFilterSize = 3)
    
    #@ save the preprocessed data
    #trainDataPP.savePPData(dirName = "pp_data2")

    #@ load preprocessed data in txt created by euijun
    trainDataPP2 = dataLoader('./euijun/data/prices/*.txt', type = "pptxt")
    trainDataPP2.load()

    #@ load raw test data
    testDataSet = []
    for idx in range(0, 9 + 1):
        testDataSet.append(dataLoader(f'./aT_test_raw/sep_{idx}/pummok_*.csv', saveFilePrefix = f"test{idx}_", type = "csv"))
        testDataSet[idx].load(save = False, load = True)    #@ switch save & load arguments when in need

    #@ plot raw data and euijun's preprocessed data
    for plot_type in range(26, 27):
        plt.figure(figsize = (30, 18))
        plt.title(f"Daily Average Price of Type {plot_type}")
        plt.plot(trainDataRaw.pummokData[plot_type]["해당일자_전체평균가격(원)"])
        plt.plot(trainDataPP.pummokData[plot_type]["해당일자_전체평균가격(원)"])
        plt.legend(["raw", "pp"])
        plt.show()

    #@ create moving average
    trainDataPP.createMovingAvg(30)
    #@ plot raw data, euijun's preprocessed data, moving average of 30 days
    plot_type = 0
    plt.plot(trainDataRaw.pummokData[plot_type]["해당일자_전체평균가격(원)"])
    plt.plot(trainDataPP.pummokData[plot_type]["해당일자_전체평균가격(원)"])
    plt.plot(trainDataPP.mvavg[plot_type])
    plt.legend(["raw", "pp", "30"])
    plt.show()

    #@ check if the migrated preprocessing methods are the same
    # for type in range(0,37):
    #     pp1 = trainDataPP.pummokData[type]["해당일자_전체평균가격(원)"].tolist()
    #     pp2 = trainDataPP2.pummokData[type]

    #     for p1, p2 in zip(pp1, pp2):
    #         if p1!=p2:
    #             print(f"different! type: {type}")
    # print(f"equality check complete")
