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
        self.data_list = glob(dir) #@JJ glob returns files in arbitrary order
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
            sep2['dateOfWeek'] = sep2['datadate'].dt.weekday # add columt of weekdays


            # sql 문법을 이용해 '해당일자_전체거래물량' 계산

            # height_set = []
            # low_set = []
            # height_volume_set = []
            # low_volume_set = []

            # for i in sep2['datadate'][2:]:

            #     """
            #     sep2는 group by를 통해 각 일자가 합쳐진 상태 예를 들어 '201703' 이 5개 이렇게 있을때 sep2는 group 시켜서 '해당일자_전체거래물량'을 계산
            #     이후 sep2 기준 20170101 and 20220630 사이의 날짜들에 해당하는 각 '201703' 마다 '해당일자_전체평균가격' 보다 큰지 아니면 작은지 판단
            #     위 과정을 통해 '하위가격 평균가', '상위가격 평균가', '하위가격 거래물량', '상위가격 거래물량' 변수 생성
            #     """

            #     new_list = ddf.iloc[[d for d, x in enumerate(ddf['datadate']) if x == i]]


            #     set_price = sep2.loc[list(sep2['datadate']).index(i)]['해당일자_전체평균가격(원)']

            #     sum_he_as = sum(new_list['거래대금(원)'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z >= set_price)
            #     sum_he_vo = sum(new_list['거래량'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z >= set_price)

            #     sum_lo_as = sum(new_list['거래대금(원)'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z < set_price)
            #     sum_lo_vo = sum(new_list['거래량'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z < set_price)

            #     if sum_lo_vo != 0:
            #         low_set.append(sum_lo_as / sum_lo_vo)
            #         low_volume_set.append(sum_lo_vo)
            #     else:
            #         low_set.append(np.nan)
            #         low_volume_set.append(np.nan)

            #     if sum_he_vo != 0:
            #         height_set.append(sum_he_as / sum_he_vo)
            #         height_volume_set.append(sum_he_vo)
            #     else:
            #         height_set.append(np.nan)
            #         height_volume_set.append(np.nan)

            # sep2['하위가격 평균가(원)'] = low_set
            # sep2['상위가격 평균가(원)'] = height_set

            # sep2['하위가격 거래물량(kg)'] = low_volume_set
            # sep2['상위가격 거래물량(kg)'] = height_volume_set

            self.pummokData[int(name.split("_")[1].split(".")[0])] = ddf.copy()
            #globals()[f'df_{name.split("_")[1].split(".")[0]}'] = sep2.copy()

            # 중간 산출물 저장
            if save != 0:
                if os.path.exists(f'./p_data') == False:
                    os.mkdir(f'./p_data')

                if os.path.exists(f'./p_data/pummok') == False:
                    os.mkdir(f'./p_data/pummok')

                sep2.to_csv(savePath, index=False)


    # def add_dosomae(self, option=1, check=0):

    #     """
    #     check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
    #     domae, somae 데이터를 가져와서 정제하는 단계
    #     option parameter을 통한 도매, 소매 선택
    #     """

    #     if option == 1:
    #         df = self.domae
    #         text = '도매'
    #     else:
    #         df = self.somae
    #         text = '소매'

    #     for i in tqdm(df):
    #         test = pd.read_csv(i)
    #         name = i.split('/')[-1]

    #         sep = test.loc[(test['등급명'] == '상품') | (test['등급명'] == 'S과')]  # 모든 상품에 대해서 수행하지 않고 GRAD_NM이 '상품', 'S과' 만 해당하는 품목 가져옴
    #         sep = sep[['datadate', '등급명', '조사단위(kg)', '가격(원)']]

    #         sep.rename(columns={"가격(원)": "가격"}, inplace=True)

    #         sep2 = sqldf(
    #             f"select datadate, max(가격) as '일자별_{text}가격_최대(원)', avg(가격) as '일자별_{text}가격_평균(원)', min(가격) as '일자별_{text}가격_최소(원)' from sep group by datadate")

    #         globals()[f'df_{name.split("_")[1].split(".")[0]}'] = globals()[f'df_{name.split("_")[1].split(".")[0]}'].merge(sep2, how='left')

    #         # 중간 산출물 저장
    #         if check != 0:
    #             if os.path.exists(f'./data') == False:
    #                 os.mkdir(f'./data')

    #             if os.path.exists(f'./data/{text}') == False:
    #                 os.mkdir(f'./data/{text}')

    #             sep2.to_csv(f'./data/{text}/{name}', index=False)

    # def add_imexport(self,check=0):
    #     """
    #     check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
    #     imexport 데이터 관련 정제, imexport 데이터는 월별 수입수출 데이터임으로 해당 월에 같은 값을 넣어주고 없는 것에는 np.nan
    #     해당 품목에 대한 imexport 데이터가 없는 경우 np.nan으로 대체, 모든 품목의 데이터가 동일한 컬럼수를 가지기 위해 수행
    #     """

    #     imex_cd = [i.split('_')[-1].split('.')[0] for i in self.imexport]

    #     for i in tqdm(range(len(self.pummok))):

    #         cd_number = self.pummok[i].split('_')[-1].split('.')[0]
    #         file_name = 'imexport_' + self.pummok[i].split('pummok_')[1]


    #         if cd_number in imex_cd:
    #             test4 = pd.read_csv(self.imexport[imex_cd.index(cd_number)])

    #             new_exim1 = []
    #             new_exim2 = []
    #             new_exim3 = []
    #             new_exim4 = []
    #             new_exim5 = []

    #             for j in globals()[f'df_{cd_number}']['datadate']:
    #                 target = j//100

    #                 try:
    #                     number = list(test4['datadate']).index(target)


    #                     new_exim1.append(test4['수출중량(kg)'].iloc[number])
    #                     new_exim2.append(test4['수출금액(달러)'].iloc[number])
    #                     new_exim3.append(test4['수입중량(kg)'].iloc[number])
    #                     new_exim4.append(test4['수입금액(달러)'].iloc[number])
    #                     new_exim5.append(test4['무역수지(달러)'].iloc[number])
    #                 except:
    #                     new_exim1.append(np.nan)
    #                     new_exim2.append(np.nan)
    #                     new_exim3.append(np.nan)
    #                     new_exim4.append(np.nan)
    #                     new_exim5.append(np.nan)

    #             df2 = pd.DataFrame()
    #             df2['수출중량(kg)'] = new_exim1
    #             df2['수출금액(달러)'] = new_exim2
    #             df2['수입중량(kg)'] = new_exim3
    #             df2['수입금액(달러)'] = new_exim4
    #             df2['무역수지(달러)'] = new_exim5

    #             globals()[f'df_{cd_number}'] = pd.concat([globals()[f'df_{cd_number}'], df2],axis=1)

    #         else:
    #             df2 = pd.DataFrame()
    #             df2['수출중량(kg)'] = np.nan
    #             df2['수출금액(달러)'] = np.nan
    #             df2['수입중량(kg)'] = np.nan
    #             df2['수입금액(달러)'] = np.nan
    #             df2['무역수지(달러)'] = np.nan

    #             globals()[f'df_{cd_number}'] = pd.concat([globals()[f'df_{cd_number}'], df2], axis=1)


    #         if check != 0:
    #             if os.path.exists(f'./data') == False:
    #                 os.mkdir(f'./data')

    #             if os.path.exists(f'./data/수출입') == False:
    #                 os.mkdir(f'./data/수출입')

    #             df2.to_csv(f'./data/수출입/{file_name}', index=False)

    # def add_weather(self, check=0):

    #     """
    #     check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
    #     weather 품목별 주산지 데이터를 가져와 합치는 함수, 일부 품목의 주산지가 3개가 아닌 것에 대해서는 np.nan 값으로 합쳐줌
    #     """

    #     for i in tqdm(self.pummok):
    #         name = i.split('_')[-1].split('.')[0]
    #         check_file = [j for j in self.weather if j.split('_')[-2] == name]


    #         df = pd.DataFrame()
    #         for d, j in enumerate(check_file):
    #             weather_df = pd.read_csv(j)
    #             new_exim1, new_exim2, new_exim3, new_exim4, new_exim5, new_exim6 = [], [], [], [], [], []


    #             for k in globals()[f'df_{name}']['datadate']:
    #                 try:
    #                     number = list(weather_df['datadate']).index(k)

    #                     new_exim1.append(weather_df['초기온도(℃)'].iloc[number])
    #                     new_exim2.append(weather_df['최대온도(℃)'].iloc[number])
    #                     new_exim3.append(weather_df['최저온도(℃)'].iloc[number])
    #                     new_exim4.append(weather_df['평균온도(℃)'].iloc[number])
    #                     new_exim5.append(weather_df['강수량(ml)'].iloc[number])
    #                     new_exim6.append(weather_df['습도(%)'].iloc[number])
    #                 except:
    #                     new_exim1.append(np.nan)
    #                     new_exim2.append(np.nan)
    #                     new_exim3.append(np.nan)
    #                     new_exim4.append(np.nan)
    #                     new_exim5.append(np.nan)
    #                     new_exim6.append(np.nan)


    #             df[f'주산지_{d}_초기온도(℃)'] = new_exim1
    #             df[f'주산지_{d}_최대온도(℃)'] = new_exim2
    #             df[f'주산지_{d}_최저온도(℃)'] = new_exim3
    #             df[f'주산지_{d}_평균온도(℃)'] = new_exim4
    #             df[f'주산지_{d}_강수량(ml)'] = new_exim5
    #             df[f'주산지_{d}_습도(%)'] = new_exim6

    #         if len(check_file) < 3:
    #             df[f'주산지_2_초기온도(℃)'] = np.nan
    #             df[f'주산지_2_최대온도(℃)'] = np.nan
    #             df[f'주산지_2_최저온도(℃)'] = np.nan
    #             df[f'주산지_2_평균온도(℃)'] = np.nan
    #             df[f'주산지_2_강수량(ml)'] = np.nan
    #             df[f'주산지_2_습도(%)'] = np.nan

    #         globals()[f'df_{name}'] = pd.concat([globals()[f'df_{name}'], df], axis=1)

    #         if check !=0:
    #             if os.path.exists(f'./data') == False:
    #                 os.mkdir(f'./data')

    #             if os.path.exists(f'./data/주산지') == False:
    #                 os.mkdir(f'./data/주산지')

    #             df.to_csv(f'./data/주산지/weather_{name}.csv', index=False)

    # def add_categorical(self, out_dir, data_type="train", check=0):

        # """
        # check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        # 일자별 정보를 넣어주는 함수, 월별, 상순, 하순, 중순 을 원핫 인코딩을 통해 데이터로 넣어주는 함수
        # 모델이 각 행마다의 정보에서 몇월인지 상순인지 하순인지 파악하며 훈련시키기 위한 변수
        # """

        # for i in tqdm(self.pummok):
        #     name = i.split('_')[-1].split('.')[0]

        #     day_set = []
        #     month_set = []

        #     for k in globals()[f'df_{name}']['datadate']:
        #         day = k % 100
        #         month = k % 10000 // 100

        #         if day <= 10:
        #             day_set.append('초순')
        #         elif (day > 10) and (day <= 20):
        #             day_set.append('중순')
        #         else:
        #             day_set.append('하순')

        #         month_set.append(f'{month}월')

        #     globals()[f'df_{name}']['일자구분'] = day_set
        #     globals()[f'df_{name}']['월구분'] = month_set

        #     globals()[f'df_{name}'] = pd.get_dummies(globals()[f'df_{name}'], columns=['일자구분', '월구분'])

        #     if check !=0:
        #         if os.path.exists(f'./data') == False:
        #             os.mkdir(f'./data')

        #         if data_type != "train":
        #             if os.path.exists(f'./data/{data_type}') == False:
        #                 os.mkdir(f"./data/{data_type}")
        #             if os.path.exists(f'./data/{data_type}/{out_dir}') == False:
        #                 os.mkdir(f'./data/{data_type}/{out_dir}')
        #             globals()[f'df_{name}'].to_csv(f'./data/{data_type}/{out_dir}/{data_type}_{name}.csv', index=False)
        #         else:
        #             if os.path.exists(f'./data/{out_dir}') == False:
        #                 os.mkdir(f'./data/{out_dir}')
        #             globals()[f'df_{name}'].to_csv(f'./data/{out_dir}/{data_type}_{name}.csv', index=False)

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
    

    # NaN analysis code
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