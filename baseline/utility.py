from sqlite3 import threadsafety
from tqdm import tqdm

import re
import numpy as np
import tensorflow as tf

## 입력 shape 및 형태 정의 함수
def make_Tensor(array):
    return tf.convert_to_tensor(array, dtype=tf.float32)

def astype_data(data, normalize=False):
    df = data.astype(np.float32)
    if normalize:
        df, _ = normalize_xy(xdata=df)
    return make_Tensor(df)

## 시점 윈도우 생성 함수
def time_window(df, t, t_sep):
    seq_len = t
    seqence_length = seq_len + t_sep

    result = []
    for index in tqdm(range(len(df) - seqence_length)):
        result.append(df[index: index + seqence_length].values)

    return np.array(result, dtype=np.float32)
        
def fill_zeros_xy(xdata, ydata):
    for j in range(len(xdata)):
        for i in range(np.shape(xdata)[1]-33):#날씨, 월구분, 중순은 뺐음. 
            col = xdata[j, :, i]
            mean = col[np.nonzero(col)].mean()
            if not np.isnan(mean):
                col[col == 0] = mean 
                xdata[j, :, i] = col
                # print("xmean: ", mean)

        col = ydata[j]
        mean = col[np.nonzero(col)].mean()
        if not np.isnan(mean):
            col[col == 0] = mean
            ydata[j] = col
            # print("ymean: ", mean) 

    return xdata, ydata

def normalize_xy(xdata, ydata=[], idx=0):
    idx_to_remove = []
    if len(ydata)!=0: 
        threshold = 5
    else:
        threshold = 2

    for i in range(len(xdata)):
        for j in range(np.shape(xdata[i])[1]-33): #날씨, 월구분, 중순은 뺐음. 
            p = xdata[i, :, j]
            if len(np.unique(p)) < threshold:
                xdata[i, :, j]=np.zeros_like(xdata[i, :, j])
                if j == idx:
                    idx_to_remove.append(i)
            else:
                last = p[p!=0][-1]
                xdata[i, :, j] = (p - last)/last
                if j == idx and len(ydata)!=0:
                    q = ydata[i]
                    if len(np.unique(q)) < threshold:
                        idx_to_remove.append(i)
                        q = np.zeros_like(ydata[i])
                    else:
                        ydata[i] = (q - last)/last
 

    for i in sorted(idx_to_remove, reverse=True):
        if len(ydata)!=0:
            xdata = np.delete(xdata, i, axis=0)
            ydata = np.delete(ydata, i, axis=0)
        # print(f"deleted {i}")

    # print(xdata[100, :, idx])
    # print(ydata[100])

    return xdata, ydata

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text