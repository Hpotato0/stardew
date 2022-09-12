from glob import glob
import utility

## 데이터 불러오기 및 parameter 설정

def fetch_data_list(path='./data/train/*.csv'):
    data_list_original = glob(path)
    data_list_original.sort(key=utility.natural_keys)
    data_list = []
    data_list_without_imexport = []
    

    for i in range(len(data_list_original)):
        if i not in without_imexport:
            data_list.append(data_list_original[i])
        else:
            data_list_without_imexport.append(data_list_original[i])

    return data_list_original, data_list, data_list_without_imexport

without_imexport = [7, 8, 14, 16, 23, 24, 26, 27, 31]
# without_imexport = [7, 16]
data_list_original, data_list, data_list_without_imexport = fetch_data_list()
tr_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 '] # train 에서 사용하지 않는 열
ts_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 '] # test 에서 사용하지 않는 열
check_col = ['일자구분_중순', '일자구분_초순', '일자구분_하순','월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', 
            '월구분_4월','월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월'] # 열 개수 맞추기
