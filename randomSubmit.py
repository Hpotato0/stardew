"""
Create a csv file in answer format filled with random numbers of normal distribution
"""

import pandas as pd
import random as random
import numpy as np

out_ans = pd.read_csv('./answer_example.csv')
for col in out_ans:
    if col != "Set":
        for idx, e in enumerate(out_ans[col]):
            if e == 0.0:
                out_ans[col][idx] = np.random.normal(0,0.001)#random.uniform(-0.001, 0.001)

out_ans.to_csv('./submit_zzb5.csv',index=False)