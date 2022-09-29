from preprocessing import preprocessing_data
from scipy.stats import pearsonr   
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    data = preprocessing_data(dir='../aT_train_raw/pummok_*.csv', 
                              dir_outlier='./outliers')

    data.chaboon()
    data.moving_avg(100)
    data.chaboon(mvavg=1)

    corr = []
    corr_diff = []
    corr_mvavg = []
    corr_mvavg_diff = []

    for i in range(len(data)):
        splits = np.split(data.prices[i], [365, 365*2, 365*3, 365*4])
        splits.pop(-1)
        splits_diff = np.split(data.prices_diff[i], [365, 365*2, 365*3])
        splits_mvavg = np.split(data.prices_mvavg[i], [365, 365*2, 365*3, 365*4])
        splits_mvavg.pop(-1)
        splits_mvavg_diff = np.split(data.prices_mvavg_diff[i], [365, 365*2, 365*3])
        
        # 변수이름 거지같이 만듦
        corrs = []
        corrs_diff = []
        corrs_mvavg = []
        corrs_mvavg_diff = []

        for j in range(len(splits)):
            for k in range(j+1, len(splits)):
                corrs.append(pearsonr(splits[j], splits[k])[0])
                corrs_diff.append(pearsonr(splits_diff[j], splits_diff[k])[0])
                corrs_mvavg.append(pearsonr(splits_mvavg[j], splits_mvavg[k])[0])
                corrs_mvavg_diff.append(pearsonr(splits_mvavg_diff[j], splits_mvavg_diff[k])[0])

        # print(i, min(corrs), min(corrs_diff), min(corrs_mvavg), min(corrs_mvavg_diff))
        # 일단은 최소로 했는데, 재형이가 말했듯이 끝에서 두번째로 해도 될 듯? 아닌가? ㅁㄹ
        corr.append(min(corrs))
        corr_diff.append(min(corrs_diff))
        corr_mvavg.append(min(corrs_mvavg))
        corr_mvavg_diff.append(min(corrs_mvavg_diff))

    whatthefuck = list(enumerate(corr_mvavg))
    whatthefuck_diff = list(enumerate(corr_mvavg_diff))

    whatthefuck.sort(key=lambda x:x[1])
    whatthefuck_diff.sort(key=lambda x:x[1])

    # correlation이 높은 상위 5개의 품목은?! (각각 moving average, moving average의 차분으로 계산)
    # print(whatthefuck[-5:])
    # print(whatthefuck_diff[-5:])

    # 11번이 두 분야 모두에서 1등을 차지했네요. 

    print(whatthefuck)
    print(whatthefuck_diff)

    plt.plot(data.prices[22])
    plt.show()

    # plt.plot(data.prices_mvavg[17])
    # plt.plot(data.prices_diff[25])
    # plt.show()