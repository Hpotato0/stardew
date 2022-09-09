#@ a simpler version of preprocessor by HPotato0
#@ loads pummok.csv data & plots price by time
#@ code for seeking trends per weekday/extract NaN periods are commented out
#@ NO USE NOW EXCEPT FOR PLOTTING.. USE CLASS IN loadData.py FOR SIMPLER PREPROCESSING
#@ THIS CODE WILL SAVE/LOAD PREPROCESSED DATA FROM ./p_data/pummok

from glob import glob
from tqdm import tqdm
from pandasql import sqldf
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

## 전처리 Class
class j_preprocessing_data(object):

    """
    도매, 소매, 수입수출, 도매경락, 주산지 데이터 전처리용 class
    중간결과물 저장 check parameter을 통해 지정, 중간결과물 저장 없이 사용은 check = 0
    """

    def __init__(self, dir, saveFilePrefix = ""):
        """
        전체 데이터에서 해당하는 domae,imexport,pummok,somae,weather 별 분리
        """
        
        self.domae = []
        self.imexport = []
        self.pummok = []
        self.somae = []
        self.weather = []

        for i in self.data_list:
            if 'domae' in i:
                self.domae.append(i)
            if 'imexport' in i:
                self.imexport.append(i)
            if 'pummok' in i.split('/')[-1]: #@JJ Why?
                self.pummok.append(i)
            if 'somae' in i:
                self.somae.append(i)
            if 'weather' in i:
                self.weather.append(i)

        self.pummokData = [None]*(37)
        self.saveFilePrefix = saveFilePrefix


    def add_pummock(self, save = 0, load = 0):

        """
        check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        pummock의 데이터를 가져와 '해당일자_전체거래물량', '하위가격 평균가', '상위가격 평균가', '하위가격 거래물량', '상위가격 거래물량' 의 파생변수를 생성하는 단계
        """

        for filepath in tqdm(self.pummok):
            name = filepath.split('/')[-1] # 전체 정제한 데이터를 담을 변수 이름
            savePath = f"./p_data/pummok/{self.saveFilePrefix}{name}"

            if load != 0:
                ddf = pd.read_csv(savePath, dtype = {"datadate": str, "단가(원)": float, "거래량": float, "해당일자_전체평균가격(원)": float, "해당일자_전체거래물량(ton)": float, "dateOfWeek": 'Int8'})
                self.pummokData[int(name.split("_")[1].split(".")[0])] = ddf.copy()
                #globals()[f'df_{name.split("_")[1].split(".")[0]}'] = ddf.copy()
                continue

            #ddf = pd.read_csv(filepath, dtype = {"datadate": str, "단가(원)": float, "거래량": float, "거래대금(원)": float, "경매건수": float, "도매시장코드": 'category', "도매법인코드": 'category', "산지코드 ": 'category', "해당일자_전체평균가격(원)": float})  # pummock의 csv 읽어오기
            ddf = pd.read_csv(filepath, usecols = ["datadate", "단가(원)", "거래량", "해당일자_전체평균가격(원)"], dtype = {"datadate": str, "단가(원)": float, "거래량": float, "거래대금(원)": float, "경매건수": float, "도매시장코드": 'category', "도매법인코드": 'category', "산지코드 ": 'category', "해당일자_전체평균가격(원)": float})  # pummock의 csv 읽어오기
            

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
    trainData = j_preprocessing_data('./aT_train_raw/*.csv', saveFilePrefix = "train_")
    trainData.add_pummock(load = 1)

    testDataSet = []
    for idx in range(0, 9 + 1):
        testDataSet.append(j_preprocessing_data(f'./aT_test_raw/sep_{idx}/*.csv', saveFilePrefix = f"test{idx}_"))
        testDataSet[idx].add_pummock(load = 1)

    #@@ see plot
    #fig = plt.figure(figsize = (15,15))
    for typeIdx in range(0, 36 + 1):
        price = trainData.pummokData[typeIdx]['해당일자_전체평균가격(원)'].tolist()
        date = trainData.pummokData[typeIdx]['datadate'].tolist()
        x = [datetime.strptime(d, "%Y-%m-%d").date() for d in date]
        
        #fig.add_subplot(5, 8, typeIdx+1)
        plt.plot(x, price)
        plt.plot(x[:14], np.ones(14) * 3000, c = 'red')
        #plt.axis('off')
        plt.title(typeIdx)
        plt.show()
    #plt.show()

    # weekDays = ['Wed', "Thu", 'Fri', 'Sat', 'Sun', 'Mon']
    # weekSepDataSet = []
    # for typeIdx, df in enumerate(trainData.pummokData):
    #     weekSepDataSet.append([])
    #     weekSep = []

    #     dowIdx = df.columns.get_loc('dateOfWeek')
    #     for idx, val in enumerate(df['해당일자_전체평균가격(원)']):
    #         if df.iloc[idx, dowIdx] == 1:
    #             if len(weekSep) == 6:
    #                 weekSepDataSet[typeIdx].append(weekSep)
    #             weekSep = []
    #             continue
    #         weekSep.append(val)
    #     if len(weekSep) == 6:
    #         weekSepDataSet[typeIdx].append(weekSep)

    # for typeIdx in range(0, 36 + 1):
    #     for weekSep in weekSepDataSet[typeIdx]:
    #         n = weekSep #np.log( np.array(weekSep) / weekSep[0] )
    #         plt.plot(weekDays, n, c = 'red', alpha = 0.5)
    #     plt.title(typeIdx)
    #     #plt.ylim(-1, 1)
    #     plt.show()

    # for typeIdx in range(0, 36+1):
    #     w = np.array(weekSepDataSet[typeIdx])
    #     w = np.nan_to_num(w)
    #     avg = np.mean(w, axis = 0)
    #     avg = avg / avg[0]
    #     plt.plot(weekDays, avg)
    #     plt.title(typeIdx)
    #     plt.show()

    # for weekSep in weekSepDataSet[0]:
    #     plt.plot(weekDays, weekSep)
    

    #@ NaN analysis code
    # for setIdx, testData in enumerate(testDataSet):
    #     cnts = []
    #     dates = []
    #     for k, df in enumerate(testData.pummokData): #trainData.pummokData
    #         dates.append([])
    #         dateIdx = df.columns.get_loc('datadate')
    #         cache = []
    #         cnt = 0
    #         for idx, val in enumerate(df['해당일자_전체평균가격(원)']):
    #             if np.isnan(val):
    #                 datadate = datetime.strptime(df.iloc[idx, dateIdx], "%Y-%m-%d")
    #                 if cache:
    #                     if (datadate - cache[-1]).days == 1:
    #                         cache.append(datadate)
    #                     else:
    #                         if len(cache) > 1:
    #                             cnt += 1
    #                             dates[k].append(cache[0].strftime("%Y%m%d") + "-" + cache[-1].strftime("%Y%m%d"))
    #                         cache = [datadate]
    #                 else:
    #                     cache.append(datadate)
    #         if len(cache) > 1:
    #             cnt += 1
    #             dates[k].append(cache[0].strftime("%Y%m%d") + "-" + cache[-1].strftime("%Y%m%d"))
    #         cnts.append(cnt)
    #     #print(cnts)
    #     print(f"test set {setIdx}")
    #     for idx, date in enumerate(dates):
    #         if cnts[idx] != 0:
    #             print(f"{idx}: {cnts[idx]}")
    #             if date:
    #                 for d in date:
    #                     print(d + ", ", end = "")
    #                 print()
    #     print()