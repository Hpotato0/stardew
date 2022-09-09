import numpy as np
import pandas as pd

#@ Converts list of answers(change rate) in format
#@ [[[set0-type0],...,[set0-type36]],...,[[set9-type0],...,[set9-type36]]]
#@ into answer csv file
def ratelistToCsv(allList, csvName):
    #@ concat the data of each set
    for setIdx, setList in enumerate(allList):
        if setIdx == 0:
            setConcatList = setList
        else:
            setConcatList = [x + y for x, y in zip(setConcatList, setList)]

    #@ save the concatenated data
    outDf = pd.read_csv('./answer_example.csv')
    for typeIdx in range(36 + 1):
        colName = f'품목{typeIdx} 변동률'
        outDf.loc[:,colName] = outDf.loc[:,colName] + setConcatList[typeIdx] #@ NaN + number = NaN
    outDf.to_csv(f'./{csvName}.csv', index = False)

#@ Converts list of price for days [t, t+1, t+2, ..., t+28]
#@ into rate of change of the answer format
#@ i.e. the change from day 't', for days [t+1, t+2, ... t+14, t+22-t+28]
def priceToRate(priceList):
    base = priceList[0]
    rateList = [(p/base)-1 for p in priceList]
    avg = sum(rateList[22:]) / 7
    return rateList[1:15] + [avg]

#@ Converts list of price for days [t, t+1, t+2, ..., t+28] in format
#@ [[[set0-type0],...,[set0-type36]],...,[[set9-type0],...,[set9-type36]]]
#@ into answer csv file
def pricelistToCsv(allList, csvName):
    #@ convert price to rate
    rAllList = [list(map(priceToRate, setList)) for setList in allList]
    
    #@ save rate list into csv
    ratelistToCsv(rAllList, csvName)



#@ ratelistToCsv sanity check code
if __name__ == "__main__":
    #@ create dummy data in int format SSTTDD (S:set, T:type, D:day)
    d = np.array(range(1,16))
    allList = []
    for s in range(10):
        setList = [d+100*t for t in range(37)]
        allList.append((np.array(setList)+10000*s).tolist())

    #@ save the dummy data in submit format csv
    ratelistToCsv(allList, "submit_format_check")
