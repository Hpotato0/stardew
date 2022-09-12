import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

from lists import data_list_original, tr_del_list, ts_del_list, check_col

## 정답 제출 파일생성
for k in tqdm(range(10)):
    globals()[f'set_df_{k}'] = pd.DataFrame()
    answer_df_list = glob(f'./model_output/set_{k}/*.csv') # 예측한 결과 불러오기
    pum_list = glob(f'./aT_test_raw/sep_{k}/*.csv') # 기존 test input 불러오기
    pummok = [a for a in pum_list if 'pummok' in a.split('/')[-1]]

    for i in answer_df_list:
        df = pd.read_csv(i)
        number = i.split('_')[-1].split('.')[0]

        base_number = 0
        for p in pummok:
            if number == p.split('_')[-1].split('.')[0]:
                pum_df = pd.read_csv(p)

                if len(pum_df) != 0:
                    base_number = pum_df.iloc[len(pum_df)-1]['해당일자_전체평균가격(원)']  # 기존 각 sep 마다 test input의 마지막 target 값 가져오기 (변동률 계산을 위해)
                else:
                    base_number = np.nan

        globals()[f'set_df_{k}'][f'품목{number}']  = [base_number] + list(df[df.columns[-1]].values) # 각 품목당 순서를 t, t+1 ... t+28 로 변경

    globals()[f'set_df_{k}'] = globals()[f'set_df_{k}'][[f'품목{col}' for col in range(len(data_list_original))]] # 열 순서를 품목0 ~ 품목36 으로 변경

"""- 변동률 계산을 위한 t, t+1 ... t+28 설정"""

# print(globals()['set_df_0'])

"""- 변동률 계산 """

date = [f'd+{i}' for i in range(1,15)] + ['d+22 ~ 28 평균']

for k in range(10):
    globals()[f'answer_df_{k}'] = pd.DataFrame()
    for c in globals()[f'set_df_{k}'].columns:
        # base_d = globals()[f'set_df_{k}'][c][0] # 변동률 기준 t 값
        

        ans_1_14 = []
        for i in range(14):
            # ans_1_14.append((globals()[f'set_df_{k}'][c].iloc[i+1]- base_d)/base_d)  # t+1 ~ t+14 까지는 (t+n - t)/t 로 계

            ans_1_14.append(globals()[f'set_df_{k}'][c].iloc[i+1])

        # ans_22_28 = (globals()[f'set_df_{k}'][c][22:29].mean() - base_d)/base_d # t+22 ~ t+28은 np.mean(t+22 ~ t+28) - t / t
        ans_22_28 = globals()[f'set_df_{k}'][c][22:29].mean()

        globals()[f'answer_df_{k}'][f'{c} 변동률'] = ans_1_14 + [ans_22_28]
    
    globals()[f'answer_df_{k}']['Set'] = k # set 번호 설정
    globals()[f'answer_df_{k}']['일자'] = date # 일자 설정


# 위에서 계산된 변동률 들을 합쳐주는 과정

all_df =pd.DataFrame()
for i in range(10):
  if i== 0 :
    all_df = pd.concat([all_df, globals()[f'answer_df_{i}']],axis=1)
  else:
    all_df = pd.concat([all_df, globals()[f'answer_df_{i}']])


all_df = all_df[['Set','일자'] + list(all_df.columns[:-2])]
all_df.reset_index(drop=True, inplace=True)

"""- 정답 양식으로 변경"""

# set, 일자 기억하기위해 따로 저장

re_set = list(all_df['Set'])
re_date = list(all_df['일자'])


# 정답 양식 불러오기
out_ans = pd.read_csv('./answer_example.csv')

# 두 dataframe 합치기 (nan + 숫자 = nan 이용)
submit_df = all_df + out_ans

submit_df['Set'] = re_set
submit_df['일자'] = re_date


# 최종 저장
submit_df.to_csv('./submit.csv',index=False)

