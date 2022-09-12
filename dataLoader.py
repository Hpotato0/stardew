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
        self.pummokData = [None]*(37) #@ placeholder for each type's data, in correct order
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
    def ejsPreProcess(self):
        for tidx, ddf in enumerate(self.pummokData):
            price = ddf["해당일자_전체평균가격(원)"].to_numpy()
            print(len(ddf.loc[:,"해당일자_전체평균가격(원)"]))  

            #@ manual removal of outliers
            if tidx==0:
                ddf.loc[ddf["datadate"]=="20130102","해당일자_전체평균가격(원)"] = np.nan
            if tidx == 1:
                ddf.loc[ddf["datadate"]=="20140211","해당일자_전체평균가격(원)"] = np.nan
                ddf.loc[ddf["datadate"]=="20150901","해당일자_전체평균가격(원)"] = np.nan

            #@ remove outliers using median filter
            nz = np.reshape(np.argwhere(~np.isnan(price)), (1,-1))[0]
            price[nz] = ndimage.median_filter(price[nz], size = 3)
            
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

            print(len(ddf.loc[:,"해당일자_전체평균가격(원)"]))    
            ddf.loc[:,"해당일자_전체평균가격(원)"] = price




if __name__ == "__main__":


    

    

    

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