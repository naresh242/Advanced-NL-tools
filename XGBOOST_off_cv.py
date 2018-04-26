

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import dask
import dask.dataframe as dd
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import os
import itertools
from sklearn.metrics import roc_auc_score
import datetime
import pickle
import sys



#helps to serialize data to disk after training
def writeresults(path,grid_results,model,algo):  
   
    pickle.dump(model,open(path+"/"+algo+"_model.pickle", "wb"))
    pickle.dump(grid_results,open(path+"/"+algo+"_grid_results.pickle", "wb"))
    
    #file = open(path+"/"+"accfile","w") 
    print("results stored",path)
    

#validates the testdata with metrics Logloss,accuracy , true positive rate , false positive rate
def valmodel(model,x_train,y_train,x_test,y_test):
    y_train_prob=model.predict_proba(x_train)[:,1]
    y_train_pred=model.predict(x_train)
    y_test_prob=model.predict_proba(x_test)[:,1]
    y_test_pred = model.predict(x_test)
    
    confmat_test=confusion_matrix(y_test,y_test_pred,labels=[0,1])
    (acc,tpr,tnr,precession,talerts,calerts,falerts,tnalerts,cnalerts,fnalerts)=conmetrics(confmat_test)
    
  
    scale_pos_weight,n_estimators,learning_rate,max_depth,subsample,colsample_bytree,colsample_bylevel,scale,estop
    cv_auc_train = roc_auc_score(y_train, y_train_prob)
    cv_auc_test=roc_auc_score(y_test, y_test_prob)
    cv_gini_norm_test = auc_to_gini_norm(cv_auc_test)
    cv_score_test = [cv_auc_test, cv_gini_norm_test]
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_loglos_train=log_loss(y_train, y_train_prob,labels=[0,1])
    cv_loglos_test=log_loss(y_test, y_test_prob,labels=[0,1])
    print("***************train metrics***************************")
    print("cv_loglos_train :",cv_loglos_train)
    #print("cv_auc_train :",cv_auc_train)
    confmat_train=confusion_matrix(y_train,y_train_pred,labels=[0,1])
    conmetrics(confmat_train)
    
    print("********************test metrics************************")
    print("cv_loglos_test :",cv_loglos_test)
    #print("cv_auc_test :",cv_auc_test)
    
    
    
    
    print("***********end ***********")
    #return cv_loglos_test,acc,cv_auc_test
    return cv_loglos_test,acc

#calcualtes metrics of confution matrix
def conmetrics(conmat):
    talerts=0
    calerts=0
    falerts=0
    tnalerts=0
    cnalerts=0
    fnalerts=0

    talerts=conmat[1][0]+conmat[1][1]
    tnalerts=conmat[0][0]+conmat[0][1]
    calerts=conmat[1][1]
    cnalerts=conmat[0][0]
    falerts=conmat[0][1]
    fnalerts=conmat[1][0]
    tpr=-1
    tnr=-1
    acc=-1
    precession=-1
    acc=(float(calerts)+float(cnalerts))/(float(talerts)+float(tnalerts))
    if talerts>0:
        tp=float(calerts)/float(talerts)
    if calerts+falerts>0:
        precession=float(calerts)/(float(calerts)+float(falerts))
    if tnalerts>0:
        tnr=float(cnalerts)/float(tnalerts)
    print("talerts ",talerts)
    print("calerts ",calerts)
    print("falerts ",falerts)
    print("tnalerts ",tnalerts)
    print("cnalerts ",cnalerts)
    print("fnalerts ",fnalerts)
    print("tpr ",tpr)
    print("tnr ",tnr)
    print("precession ",precession)
    print("accuracy ",acc)
    return (acc,tpr,tnr,precession,talerts,calerts,falerts,tnalerts,cnalerts,fnalerts)




#Generate column names 
def gencol():
    mincol=['TOT_TRAFFIC_BITS', 'TOT_TRAFFIC_PKTS','CHARGEN_BITS', 'CHARGEN_PKTS','DNS_BITS', 'DNS_PKTS','DNS_AMP_BITS', 'DNS_AMP_PKTS','ICMP_BITS', 'ICMP_PKTS','MSSQL_BITS', 'MSSQL_PKTS','NTP_BITS', 'NTP_PKTS','SNMP_BITS', 'SNMP_PKTS','SSDP_BITS', 'SSDP_PKTS', 'TCP_NULL_PKTS','TCP_RST_PKTS','TCP_SYN_PKTS','UDP_BITS', 'UDP_PKTS','IPV4_PROT_BITS', 'IPV4_PROT_PKTS','TCP_BITS', 'TCP_PKTS','TOT_COUNT','UNQ_SIP_COUNT','CHARGEN_COUNT','DNS_COUNT','ICMP_COUNT','MSSQL_COUNT','NTP_COUNT','SNMP_COUNT','SSDP_COUNT','TCP_NULL_COUNT','TCP_RST_COUNT','TCP_SYN_COUNT','UDP_COUNT','UDP_SIP_COUNT','PROT_NULL_COUNT','TCP_COUNT','TCP_SIP_COUNT']
    mincol7=[(str(x)+y)for x in range(7,0,-1) for y in mincol]



    bicol=["IPv4_ADDR","CUST_ID"]
    bicol+=mincol7

    t_col=['TOTAL_BPS', 'TOTAL_PPS', 'CHARGEN_BPS', 'CHARGEN_PPS', 'DNS_BPS', 'DNS_PPS', 'DNS_AMP_BPS', 'DNS_AMP_PPS', 'ICMP_BPS', 'ICMP_PPS', 'MYSQL_BPS', 'MYSQL_PPS', 'NTP_BPS', 'NTP_PPS', 'SNMP_BPS', 'SNMP_PPS', 'SSDP_BPS', 'SSDP_PPS', 'TCP_NULL_PPS', 'TCP_RST_PPS', 'TCP_SYN_PPS', 'UDP_BPS', 'UDP_PPS', 'IPV4_BPS', 'IPV4_PPS', 'TCP_BPS', 'TCP_PPS']

    t_col=["t_"+x for x in t_col]

    bicol+=t_col

    concol=['HostDetection', 'SeverityDetection', 'FastFloodDetection']

    concol=["t_"+x for x in concol]

    bicol+=concol

    alcol=["A_ACTUAL","A_AT1","A_AT2","A_AT3","A_AT4","A_AT5", "A_AT10","A_AT20", "A_CORR","A_CT1","A_CT2","A_CT3","A_CT4","A_CT5","A_CT10","A_CT20"]
    bicol+=alcol
    
    return bicol


#this gives tuning parameters for model . tuning parameters will be diff for diff algorithms.
#this is one place where we need to modify if we want to change algorithm
def getparam():
    learning_rate = [0.1]
    max_depth = [3,5]
    n_estimators = range(50,100, 50)

    subsample = [1.0]

    colsample_bytree = [1.0]

    colsample_bylevel = [0.8]
    scale_pos_weight=[6]
    min_child_weight=[1]
    #min_child_weight=range(1,6,2)
    objective=['binary:logistic',"reg:linear"]
    gamma=[i/10.0 for i in range(0,2)]
    #reg_alpha=[1e-5, 1e-2, 0.1, 1, 100]
    reg_alpha=[0.1]
    
    params={"learning_rate":learning_rate,"max_depth":max_depth,"n_estimators":n_estimators,"subsample":subsample,"colsample_bytree":colsample_bytree,
           "colsample_bylevel":colsample_bylevel,"scale_pos_weight":scale_pos_weight,"min_child_weight":min_child_weight,"objective":objective,"gamma":gamma,"reg_alpha":reg_alpha}
    return params
   
timestamp=str(datetime.datetime.now()).replace(" ","")
resultfiledir="/home/tatacomm/srinivas/ml_results/"+"XGB_"+timestamp

os.makedirs(resultfiledir)

print("results path :",resultfiledir)

err=resultfiledir+"/err"
out=resultfiledir+"/out"

ferr = open(err, "w")
fout = open(out, "w")

original_stderr = sys.stderr
original_stdout = sys.stdout
sys.stderr = ferr
sys.stdout=fout


#reg this for limiting number of threads created in intel machine. 
os.environ["OMP_NUM_THREADS"] = "50000"
trainfile="/home/tatacomm/srinivas/data/label_latest/train/dec12_13_trainset_clean_15k.csv"
testfile="/home/tatacomm/srinivas/data/label_latest/train/dec16_testset_clean_10K.csv"

x_col=['7TOT_TRAFFIC_BITS','7TOT_TRAFFIC_PKTS','7CHARGEN_BITS','7CHARGEN_PKTS','7DNS_BITS','7DNS_PKTS','7DNS_AMP_BITS','7DNS_AMP_PKTS','7ICMP_BITS','7ICMP_PKTS','7MSSQL_BITS','7MSSQL_PKTS','7NTP_BITS','7NTP_PKTS','7SNMP_BITS','7SNMP_PKTS','7SSDP_BITS','7SSDP_PKTS','7TCP_NULL_PKTS','7TCP_RST_PKTS','7TCP_SYN_PKTS','7UDP_BITS','7UDP_PKTS','7IPV4_PROT_BITS','7IPV4_PROT_PKTS','7TCP_BITS','7TCP_PKTS','7TOT_COUNT','7UNQ_SIP_COUNT','7CHARGEN_COUNT','7DNS_COUNT','7ICMP_COUNT','7MSSQL_COUNT','7NTP_COUNT','7SNMP_COUNT','7SSDP_COUNT','7TCP_NULL_COUNT','7TCP_RST_COUNT','7TCP_SYN_COUNT','7UDP_COUNT','7UDP_SIP_COUNT','7PROT_NULL_COUNT','7TCP_COUNT','7TCP_SIP_COUNT','6TOT_TRAFFIC_BITS','6TOT_TRAFFIC_PKTS','6CHARGEN_BITS','6CHARGEN_PKTS','6DNS_BITS','6DNS_PKTS','6DNS_AMP_BITS','6DNS_AMP_PKTS','6ICMP_BITS','6ICMP_PKTS','6MSSQL_BITS','6MSSQL_PKTS','6NTP_BITS','6NTP_PKTS','6SNMP_BITS','6SNMP_PKTS','6SSDP_BITS','6SSDP_PKTS','6TCP_NULL_PKTS','6TCP_RST_PKTS','6TCP_SYN_PKTS','6UDP_BITS','6UDP_PKTS','6IPV4_PROT_BITS','6IPV4_PROT_PKTS','6TCP_BITS','6TCP_PKTS','6TOT_COUNT','6UNQ_SIP_COUNT','6CHARGEN_COUNT','6DNS_COUNT','6ICMP_COUNT','6MSSQL_COUNT','6NTP_COUNT','6SNMP_COUNT','6SSDP_COUNT','6TCP_NULL_COUNT','6TCP_RST_COUNT','6TCP_SYN_COUNT','6UDP_COUNT','6UDP_SIP_COUNT','6PROT_NULL_COUNT','6TCP_COUNT','6TCP_SIP_COUNT','5TOT_TRAFFIC_BITS','5TOT_TRAFFIC_PKTS','5CHARGEN_BITS','5CHARGEN_PKTS','5DNS_BITS','5DNS_PKTS','5DNS_AMP_BITS','5DNS_AMP_PKTS','5ICMP_BITS','5ICMP_PKTS','5MSSQL_BITS','5MSSQL_PKTS','5NTP_BITS','5NTP_PKTS','5SNMP_BITS','5SNMP_PKTS','5SSDP_BITS','5SSDP_PKTS','5TCP_NULL_PKTS','5TCP_RST_PKTS','5TCP_SYN_PKTS','5UDP_BITS','5UDP_PKTS','5IPV4_PROT_BITS','5IPV4_PROT_PKTS','5TCP_BITS','5TCP_PKTS','5TOT_COUNT','5UNQ_SIP_COUNT','5CHARGEN_COUNT','5DNS_COUNT','5ICMP_COUNT','5MSSQL_COUNT','5NTP_COUNT','5SNMP_COUNT','5SSDP_COUNT','5TCP_NULL_COUNT','5TCP_RST_COUNT','5TCP_SYN_COUNT','5UDP_COUNT','5UDP_SIP_COUNT','5PROT_NULL_COUNT','5TCP_COUNT','5TCP_SIP_COUNT','4TOT_TRAFFIC_BITS','4TOT_TRAFFIC_PKTS','4CHARGEN_BITS','4CHARGEN_PKTS','4DNS_BITS','4DNS_PKTS','4DNS_AMP_BITS','4DNS_AMP_PKTS','4ICMP_BITS','4ICMP_PKTS','4MSSQL_BITS','4MSSQL_PKTS','4NTP_BITS','4NTP_PKTS','4SNMP_BITS','4SNMP_PKTS','4SSDP_BITS','4SSDP_PKTS','4TCP_NULL_PKTS','4TCP_RST_PKTS','4TCP_SYN_PKTS','4UDP_BITS','4UDP_PKTS','4IPV4_PROT_BITS','4IPV4_PROT_PKTS','4TCP_BITS','4TCP_PKTS','4TOT_COUNT','4UNQ_SIP_COUNT','4CHARGEN_COUNT','4DNS_COUNT','4ICMP_COUNT','4MSSQL_COUNT','4NTP_COUNT','4SNMP_COUNT','4SSDP_COUNT','4TCP_NULL_COUNT','4TCP_RST_COUNT','4TCP_SYN_COUNT','4UDP_COUNT','4UDP_SIP_COUNT','4PROT_NULL_COUNT','4TCP_COUNT','4TCP_SIP_COUNT','3TOT_TRAFFIC_BITS','3TOT_TRAFFIC_PKTS','3CHARGEN_BITS','3CHARGEN_PKTS','3DNS_BITS','3DNS_PKTS','3DNS_AMP_BITS','3DNS_AMP_PKTS','3ICMP_BITS','3ICMP_PKTS','3MSSQL_BITS','3MSSQL_PKTS','3NTP_BITS','3NTP_PKTS','3SNMP_BITS','3SNMP_PKTS','3SSDP_BITS','3SSDP_PKTS','3TCP_NULL_PKTS','3TCP_RST_PKTS','3TCP_SYN_PKTS','3UDP_BITS','3UDP_PKTS','3IPV4_PROT_BITS','3IPV4_PROT_PKTS','3TCP_BITS','3TCP_PKTS','3TOT_COUNT','3UNQ_SIP_COUNT','3CHARGEN_COUNT','3DNS_COUNT','3ICMP_COUNT','3MSSQL_COUNT','3NTP_COUNT','3SNMP_COUNT','3SSDP_COUNT','3TCP_NULL_COUNT','3TCP_RST_COUNT','3TCP_SYN_COUNT','3UDP_COUNT','3UDP_SIP_COUNT','3PROT_NULL_COUNT','3TCP_COUNT','3TCP_SIP_COUNT','2TOT_TRAFFIC_BITS','2TOT_TRAFFIC_PKTS','2CHARGEN_BITS','2CHARGEN_PKTS','2DNS_BITS','2DNS_PKTS','2DNS_AMP_BITS','2DNS_AMP_PKTS','2ICMP_BITS','2ICMP_PKTS','2MSSQL_BITS','2MSSQL_PKTS','2NTP_BITS','2NTP_PKTS','2SNMP_BITS','2SNMP_PKTS','2SSDP_BITS','2SSDP_PKTS','2TCP_NULL_PKTS','2TCP_RST_PKTS','2TCP_SYN_PKTS','2UDP_BITS','2UDP_PKTS','2IPV4_PROT_BITS','2IPV4_PROT_PKTS','2TCP_BITS','2TCP_PKTS','2TOT_COUNT','2UNQ_SIP_COUNT','2CHARGEN_COUNT','2DNS_COUNT','2ICMP_COUNT','2MSSQL_COUNT','2NTP_COUNT','2SNMP_COUNT','2SSDP_COUNT','2TCP_NULL_COUNT','2TCP_RST_COUNT','2TCP_SYN_COUNT','2UDP_COUNT','2UDP_SIP_COUNT','2PROT_NULL_COUNT','2TCP_COUNT','2TCP_SIP_COUNT','1TOT_TRAFFIC_BITS','1TOT_TRAFFIC_PKTS','1CHARGEN_BITS','1CHARGEN_PKTS','1DNS_BITS','1DNS_PKTS','1DNS_AMP_BITS','1DNS_AMP_PKTS','1ICMP_BITS','1ICMP_PKTS','1MSSQL_BITS','1MSSQL_PKTS','1NTP_BITS','1NTP_PKTS','1SNMP_BITS','1SNMP_PKTS','1SSDP_BITS','1SSDP_PKTS','1TCP_NULL_PKTS','1TCP_RST_PKTS','1TCP_SYN_PKTS','1UDP_BITS','1UDP_PKTS','1IPV4_PROT_BITS','1IPV4_PROT_PKTS','1TCP_BITS','1TCP_PKTS','1TOT_COUNT','1UNQ_SIP_COUNT','1CHARGEN_COUNT','1DNS_COUNT','1ICMP_COUNT','1MSSQL_COUNT','1NTP_COUNT','1SNMP_COUNT','1SSDP_COUNT','1TCP_NULL_COUNT','1TCP_RST_COUNT','1TCP_SYN_COUNT','1UDP_COUNT','1UDP_SIP_COUNT','1PROT_NULL_COUNT','1TCP_COUNT','1TCP_SIP_COUNT','t_TOTAL_BPS','t_TOTAL_PPS','t_CHARGEN_BPS','t_CHARGEN_PPS','t_DNS_BPS','t_DNS_PPS','t_DNS_AMP_BPS','t_DNS_AMP_PPS','t_ICMP_BPS','t_ICMP_PPS','t_MYSQL_BPS','t_MYSQL_PPS','t_NTP_BPS','t_NTP_PPS','t_SNMP_BPS','t_SNMP_PPS','t_SSDP_BPS','t_SSDP_PPS','t_TCP_NULL_PPS','t_TCP_RST_PPS','t_TCP_SYN_PPS','t_UDP_BPS','t_UDP_PPS','t_IPV4_BPS','t_IPV4_PPS','t_TCP_BPS','t_TCP_PPS','t_HostDetection','t_SeverityDetection','t_FastFloodDetection']
y_col='A_CORR'

#loading train data 
dask_df = pd.read_csv(trainfile,header=None)
#loading test data. This is future dates data
df_test=pd.read_csv(testfile,header=None)
dask_df.columns=gencol()
df_test.columns=gencol()

#this indicates which row is test and which row is validation for GridSearchCV funtion
test_fold= [-1]*dask_df.shape[0]+[0]*df_test.shape[0]
ps = PredefinedSplit(test_fold)


#merged train and test data to give for training
dataset=pd.concat([dask_df,df_test],ignore_index=True)
print("merged the data")

#assign the column names
dataset.columns=gencol()
x = dataset[x_col]
y = dataset[y_col]


param_grid=getparam()

#model initialization. this will be modified if we want to change algorithm
model = XGBClassifier(nthread=1)

#take advantage of gridsearch for parameter tuning
grid_search=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=100,cv=ps,scoring="neg_log_loss",verbose=1)
#start training
grid_result = grid_search.fit(x, y)
print (grid_result.best_score_)

model_bst=grid_result.best_estimator_

#serialize model into disk for future use
writeresults(resultfiledir,grid_result,model_bst,"XGB")
x_train = dask_df[x_col]
y_train= dask_df[y_col]

x_test = df_test[x_col]
y_test = df_test[y_col]

#get accuracy of the model
valmodel(model_bst,x_train,y_train,x_test,y_test)




