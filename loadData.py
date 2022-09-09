# THIS CODE WILL SAVE/LOAD PREPROCESSED DATA FROM ./pp_data/pummok

from glob import glob
from tqdm import tqdm
from pandasql import sqldf
from datetime import datetime
from utility import natural_keys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class pummokLoader(object):
    #@ dir: aT_train_raw/pummok_*.csv
    def __init__(self, dir, saveFilePrefix = ""):
        def __init__(self, dir, saveFilePrefix = ""):
            self.data_list = glob(dir) 

            self.pummok_list = glob(dir) #@ list of pummok csv directories, in random order
            self.pummok_list.sort(key = natural_keys)
            
            self.pummokData = [None]*(37) #@ placeholder for each type's data

            self.saveFilePrefix = saveFilePrefix

    #@ load pummok data from csv file
    #@ save/load simplified data in ./pp_data/pummok
    def add_pummock(self, save = 0, load = 0):
        #@ iterate through csvs for different types
        for filepath in tqdm(self.pummok):
            name = filepath.split('/')[-1] #@ e.g ".../pummok_5.csv" -> "pummok_5.csv"
            savePath = f"./p_data/pummok/{self.saveFilePrefix}{name}"

            if load != 0:
                ddf = pd.read_csv(savePath, dtype = {"datadate": str, "단가(원)": float, "거래량": float, "해당일자_전체평균가격(원)": float, "해당일자_전체거래물량(ton)": float, "dateOfWeek": 'Int8'})
                self.pummokData[int(name.split("_")[1].split(".")[0])] = ddf.copy()
                #globals()[f'df_{name.split("_")[1].split(".")[0]}'] = ddf.copy()
                continue

            ddf = pd.read_csv(filepath, usecols = ["datadate", "해당일자_전체평균가격(원)"], dtype = {"datadate": str, "해당일자_전체평균가격(원)": float})
            

            sep2 = sqldf(f"select *, sum(거래량) as '해당일자_전체거래물량(ton)' from ddf group by datadate")
            sep2 = sep2.fillna(value=np.nan) # replace all 'None' with 'NaN'
            sep2['datadate'] = pd.to_datetime(sep2['datadate']) # set format of "datadate" to date format
            sep2['dateOfWeek'] = sep2['datadate'].dt.weekday # add column of weekdays

            self.pummokData[int(name.split("_")[1].split(".")[0])] = ddf.copy()
            #globals()[f'df_{name.split("_")[1].split(".")[0]}'] = sep2.copy()

            # 중간 산출물 저장
            if save != 0:
                if os.path.exists(f'./p_data') == False:
                    os.mkdir(f'./p_data')

                if os.path.exists(f'./p_data/pummok') == False:
                    os.mkdir(f'./p_data/pummok')

                sep2.to_csv(savePath, index=False)

if __name__ == "__main__":
    trainData = pummokLoader('./aT_train_raw/pummok_*.csv', saveFilePrefix = "train_")
    trainData.add_pummock()