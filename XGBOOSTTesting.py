from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from pandas import read_csv
from matplotlib import pyplot
import pandas as pd
import numpy as np
import pickle

talerts=0
calerts=0
falerts=0
tnalerts=0
cnalerts=0
fnalerts=0


def cust_acc(y_test,y_pred):
    global talerts
    global calerts
    global falerts
    global tnalerts
    global cnalerts
    global fnalerts

    for i in range(len(y_pred)):
        if y_test[i]==0:
            tnalerts+=1
            if y_pred[i]==0:
                cnalerts+=1
            else:
                falerts+=1
        else:
            talerts+=1
            if y_pred[i]==1:
                calerts+=1
            else:
                fnalerts+=1
    print "###################after tested total observations ",tnalerts+cnalerts
    print "accuracy ",float(calerts+cnalerts)/float(talerts+tnalerts)
    print "total alerts ",talerts
    print "detected correctly alerts ",calerts
    print "false alerts ",falerts
    print "total non alerts ",tnalerts
    print "detected correctly non alerts ",cnalerts
    print "false non alerts ",fnalerts
    if calerts+fnalerts>0:
        print "precision ", float(calerts)/float(calerts+fnalerts) 
    if talerts>0:
        print "true positiverate(alerts) ",float(calerts)/float(talerts)
        
    if tnalerts>0:
        print "true positiverate(non alerts) ",float(cnalerts)/float(tnalerts)
                                                                     
                                                                     
modelfile="/home/tatacomm/nsrikaku/models/xgboost_roc_150k.pickle"
loaded_model = pickle.load(open(modelfile, "rb"))
lis=[1]
lis.extend(range(4,200))

#testfile="/mnt/nfs/data2/nsrikaku/combined/comday/db_df_20171001.csv"
testfile="/home/tatacomm/nsrikaku/data/labeled/clean_db_df_20171003_10-12.csv"
reader = pd.read_csv(testfile,chunksize=10000,header=None)
for chunk in reader:

    ext_test_featuredf=chunk[lis]
    ext_testdataset=ext_test_featuredf.values
    ext_X = ext_testdataset[:,1:]
    ext_Y = ext_testdataset[:,0]
    ext_pred=loaded_model.predict_proba(ext_X)
    tlabel=[]
    for x in ext_pred:
        if x[1]>0:
           tlabel.append(1)
        else:
           tlabel.append(0)
    tn=np.array(tlabel)
    cust_acc(ext_Y,tn)
    

                                                                     
