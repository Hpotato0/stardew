import warnings
import os, random
import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

from utility import astype_data
from lists import data_list, ts_del_list, check_col

# 경고 끄기
warnings.filterwarnings(action='ignore')

# 시드고정
tf.random.set_seed(19970119)
random.seed(19970119)
np.random.seed(19970119)

epoch = 10
batch = 64

## Test 과정
zero_csv = [0 for i in range(14)]  # 시점이 비어있는 데이터 0으로 채우기 위한 변수

for i in tqdm(range(10)): #원래 10임
    data_list = glob(f'./data/test/set_{i}/*.csv')
    
    for idx, j in enumerate(data_list):
        df = pd.read_csv(j)

        if len(df) == 0:
            df['zero_non'] = zero_csv
            df = df.fillna(0)
            df.drop('zero_non', axis=1, inplace=True)

        file_number = j.split('test_')[1].split('.')[0]

        # if file_number != '0':
        #     continue

        if i == 9 and file_number == '7':
            pred = tf.zeros(shape = (1, 28), dtype=tf.int32)
            save_df = pd.DataFrame(pred).T
            save_df.to_csv(f'./model_output/set_{i}/predict_{file_number}.csv', index=False)
            continue

        # 사용할 열 선택, index 설정
        df.drop(ts_del_list, axis=1, inplace=True)
        df.set_index('datadate', drop=True, inplace=True)

        # train input 과 형상 맞추기
        add_col = [i for i in check_col if i not in df.columns]

        for a in add_col:
            df[a] = 0

        # ' ' -> nan 으로 변경
        for a in df.columns:
            df[a] = df[a].replace({' ': np.nan})

        # nan 처리
        df = df.fillna(0)

        # x_test  생성
        df_test = astype_data(df.values.reshape(1, df.values.shape[0], df.values.shape[1]), normalize=True)

        # model test
        if os.path.exists('./model_output') == False:
            os.mkdir('./model_output')

        if os.path.exists(f'./model_output/set_{i}') == False:
            os.mkdir(f'./model_output/set_{i}')

        # 해당하는 모델 불러오기
        model_test = tf.keras.models.load_model(f'./model/{epoch}/transformer-{file_number}-{epoch}-{batch}.h5')
        pred = model_test.predict(df_test)

        # 결과 저장
        save_df = pd.DataFrame(pred).T
        save_df.to_csv(f'./model_output/set_{i}/predict_{file_number}.csv', index=False)

