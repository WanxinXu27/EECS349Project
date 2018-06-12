import pandas as pd
import numpy as np
import os

def fuseFeature(df, ToBeFused_list):
    n = []
    name = ""
    for feature in ToBeFused_list:
        n.append(df[feature].values.tolist())
        name += feature
        name += "_"
    addFeature = []
    for i in range(len(n[0])):
        Fuse = str(n[0][i])
        for j in range(1, len(n)):
            Fuse += " " + str(n[j][i])
        addFeature.append(Fuse)
    df[name] = addFeature
    # res = df.drop(ToBeFused_list, axis=1)    #drop the used columns
    return df

res =pd.read_csv('../data/userFeature.csv',nrows=200)
res = fuseFeature(res,['education','consumptionAbility'])
res = fuseFeature(res,['LBS','consumptionAbility'])
res = fuseFeature(res,['gender','consumptionAbility'])
res.to_csv('userFeature_fused.csv')

res2 =pd.read_csv('../data/adFeature.csv')
res2 = fuseFeature(res2,['productId', 'productType'])
res2.to_csv('adFeature_fused.csv')

# data = fuseFeature(data,['education','consumptionAbility'])
# data = fuseFeature(data,['LBS','consumptionAbility'])
# data = fuseFeature(data,['gender','consumptionAbility'])
# data = fuseFeature(data,['productId', 'productType'])
# data = fuseFeature(data,['advertiserId', 'LBS'])