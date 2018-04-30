
# coding: utf-8

# In[8]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
import time


# In[6]:


from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np


# In[7]:


def auc_to_gini_norm(auc_score):
    return 2*auc_score-1


# In[4]:


range(4)


# In[29]:


def logitml(x_train, y_train , x_test, kf):
    penalty="l2"
    bstmet=(0,0,0,0,0)
    bstmodel=None
    for C in [.5,.75,1,1.25]:
        modarr=[]
        for i in range(kf.n_splits):
            logit=LogisticRegression(random_state=1,C=C,n_jobs=30,penalty=penalty)
            modarr.append(logit)
        
        locmodel=cross_validate_sklearn(modarr,x_train,y_train,x_test, kf)
        if locmodel[4]>bstmet[4]:
            bstmet=locmodel
            bstmodel=modarr
            
            
    return (bstmet,bstmodel)
            


# In[3]:


def cross_validate_sklearn(modarr, x_train, y_train , x_test, kf,scale=False, verbose=True):
    start_time=time.time()
    
    # initialise the size of out-of-fold train an test prediction
    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    # use the kfold object to generate the required folds
    kf.split(x_train, y_train)
    
    
    for i,  in enumerate(modarr):
        # generate training folds and validation fold
        (train_index, test_index)=f.next()
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[test_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[test_index]

        # perform scaling if required i.e. for linear algorithms
        if scale:
            scaler = StandardScaler().fit(x_train_kf.values)
            x_train_kf_values = scaler.transform(x_train_kf.values)
            x_val_kf_values = scaler.transform(x_val_kf.values)
            x_test_values = scaler.transform(x_test.values)
        else:
            x_train_kf_values = x_train_kf.values
            x_val_kf_values = x_val_kf.values
            x_test_values = x_test.values
        
        # fit the input classifier and perform prediction.
        clf.fit(x_train_kf_values, y_train_kf.values)
        val_pred=clf.predict_proba(x_val_kf_values)[:,1]
        train_pred[test_index] += val_pred

        y_test_preds = clf.predict_proba(x_test_values)[:,1]
        test_pred += y_test_preds

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))
    return cv_score, train_pred,test_pred,cv_auc,cv_gini_norm


# In[ ]:




def cross_validate_xgb(params, x_train, y_train, x_test, kf, cat_cols=[], verbose=True, 
                       verbose_eval=50, num_boost_round=4000, use_rank=True):
    start_time=time.time()

    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))
    models=[]
    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        x_test_kf=x_test.copy()

        d_train_kf = xgb.DMatrix(x_train_kf, label=y_train_kf)
        d_val_kf = xgb.DMatrix(x_val_kf, label=y_val_kf)
        d_test = xgb.DMatrix(x_test_kf)

        bst = xgb.train(params, d_train_kf, num_boost_round=num_boost_round,
                        evals=[(d_train_kf, 'train'), (d_val_kf, 'val')], verbose_eval=verbose_eval,
                        early_stopping_rounds=50)

        val_pred = bst.predict(d_val_kf, ntree_limit=bst.best_ntree_limit)
        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred+=probability_to_rank(bst.predict(d_test))
        else:
            train_pred[val_index] += val_pred
            test_pred+=bst.predict(d_test)

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, 
                                                                                     fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

        return cv_score, train_pred,test_pred


# In[ ]:


scores = model_selection.cross_val_score(clf, X, y,cv=3, scoring='accuracy')


# In[4]:


df=pd.DataFrame(columns=range(100))

df


# In[9]:


#this modifies to display 500 columns
pd.set_option("display.max_columns",500)
pd.set_option("display.max_colwidth",500)
pd.set_option('display.max_rows', 1000)


# In[18]:


df=pd.DataFrame()
df["X1"]=[1,2,3,9,10,11]
df["X2"]=[1,2,3,9,10,11]
df["Y"]=[1,1,2,2,1,2]
clf=xgb.XGBClassifier()
x_train=df[["X1","X2"]]
y_train=df["Y"]
x_test=df

scale=False
verbose=True


# In[23]:


f=kf.split(x_train,y_train)


# In[24]:





# In[57]:


print df["Y"].rank()

print df["Y"].rank()/len(df["Y"])*2


# In[35]:


train_pred = np.zeros((x_train.shape[0]))
test_pred = np.zeros((x_test.shape[0]))


# In[30]:


print type(train_pred)
print train_pred.shape


# In[52]:


for i,(train_index, test_index) in enumerate(kf.split(x_train,y_train)):
    x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[test_index, :]
    y_train_kf, y_val_kf = y_train[train_index], y_train[test_index]
        
    x_train_kf_values = x_train_kf.values
    x_val_kf_values = x_val_kf.values
    x_test_values = x_test.values
    clf.fit(x_train_kf_values, y_train_kf.values)
    val_pred=clf.predict_proba(x_val_kf_values)[:,1]
    print type(val_pred)
    print (val_pred.shape)
    print val_pred
    


# In[ ]:


def cross_validate_xgb(params, x_train, y_train, x_test, kf, cat_cols=[], verbose=True, 
                       verbose_eval=50, num_boost_round=4000, use_rank=True):
    start_time=time.time()

    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        x_test_kf=x_test.copy()

        d_train_kf = xgb.DMatrix(x_train_kf, label=y_train_kf)
        d_val_kf = xgb.DMatrix(x_val_kf, label=y_val_kf)
        d_test = xgb.DMatrix(x_test_kf)

        bst = xgb.train(params, d_train_kf, num_boost_round=num_boost_round,
                        evals=[(d_train_kf, 'train'), (d_val_kf, 'val')], verbose_eval=verbose_eval,
                        early_stopping_rounds=50)

        val_pred = bst.predict(d_val_kf, ntree_limit=bst.best_ntree_limit)
        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred+=probability_to_rank(bst.predict(d_test))
        else:
            train_pred[val_index] += val_pred
            test_pred+=bst.predict(d_test)

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, 
                                                                                     fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

        return cv_score, train_pred,test_pred


# In[47]:


x_val_kf_values


# In[9]:


df=pd.DataFrame()
df["f1"]=[1,2,0,0]
df["f2"]=[1,2,3,6]
freq=df["f1"].value_counts()
freq=pd.DataFrame(freq)
freq.reset_index(inplace=True)
freq.columns=["f1","freq"]
print freq


# In[27]:


temp_train_df=pd.merge(df[["f1"]],freq,how="left",on="f1")
temp_train_df.drop(["f1"],axis=1,inplace=True) 


# In[31]:


temp_train_df.freq.astype(np.int32)


# In[21]:


ndf=df.values
n=ndf[:,1]
type(n)


# In[28]:


def crispcutter(row):
    if row>.5:
        return 1
    else:
        return 0


# In[34]:


tlabel=[]
for x in ext_pred:
    if x>0.4:
        tlabel.append(1)
    else:
        tlabel.append(0)
        


# In[36]:


tn=np.array(tlabel)
tn.shape


# In[77]:



starttime = datetime.datetime(2017,9,30,6,30)

date_time = starttime
filepat=format(date_time.strftime('%Y%m%d_%H%M'))+"(.*)\.flows"
print filepat


# In[20]:


df


# In[83]:


df[df.apply(ismatched,axis=1)]



# In[81]:


def ismatched(row):
        
    if re.match(filepat,row[2]) is not None:
        return True
    else:
        return False
    


# In[79]:



filepat="20170930_06300(.*)\.flows"

print filepat

print re.match(filepat,"20170930_063001.flows")


# In[ ]:


import sys
import os
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
from multiprocessing import Manager
import datetime
import pandas as pd
import subprocess
import json
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from generatelabelsnew import process_minute_minibatch_new


df_header="EXPORTER_IPV4_ADDRESS|IPV4_SRC_ADDR|IPV4_DST_ADDR|INPUT_SNMP|OUTPUT_SNMP|IN_PKTS|IN_BYTES|FIRST_SWITCHED|LAST_SWITCHED|L4_SRC_PORT|L4_DST_PORT|TCP_FLAGS|PROTOCOL|SRC_AS|DST_AS|IPV4_SRC_MASK|IPV4_DST_MASK|SRC_CUSTOMER|DEST_CUSTOMER"
def ismatched(row):
        
    if re.match(filepat,row[2]) is not None:
        return True
    else:
        return False
def parse_date_str(datestr):
    dlist = datestr.split(":")
    #dlist = [s.lstrip("0") for s in dlist]
    dlist = list(map(int,dlist))
    return dlist[0],dlist[1],dlist[2],dlist[3],dlist[4]

def do_job(tasks_to_accomplish, tasks_that_are_done):
    sys.stdout  = open("log/process_"+str(os.getpid())+".txt","a+")
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will
                raise queue.Empty exception if the queue is empty.
                queue(False) function would do the same task also.
            '''
            #print("Before reading")
            task = tasks_to_accomplish.get_nowait()
            #print("After reading")
            #print("length of task {}".format(len(task)))
        except queue.Empty:

            time.sleep(2)
            #break
        else:
            '''
                if no exception has been raised, add the task completion
                message to task_that_are_done queue
            '''
            #print(task)
            #print("length of task {}".format(len(task)))
            ftime = task[:16]
            task = task[16:]
            strcsv = StringIO(task)
            #tdf1 = pd.read_json(task,typ='table')
            tdf1 = pd.read_csv(strcsv)
            #print(task)

            print("lenth of sub dataframe ",len(tdf1))
            #print("Before processing ")
            year,month,date,hour,minute=parse_date_str(ftime)
            filetime=datetime.datetime(year,month,date,hour,minute)
            process_minute_minibatch_new(tdf1,filetime)
            #print("After processing ")
            sys.stdout.flush()
            #tasks_that_are_done.put("Done")
            #print(' done by ' + current_process().name)
            #time.sleep(.5)
    return True
filename="/home/tatacomm/nsrikaku/data/clean_netflow_alertedIps_6days_head"
rawMasterDf=pd.read_csv(filename,sep="|",header=None)
filepat=None
def load_3hr_data(tasks_to_accomplish,tasks_that_are_done):
    global queues
    global number_of_processes
    global rawMasterDf
    global filepat
    firsttime=True
    starttime = datetime.datetime(2017,9,29,18,1)
    endtime   = datetime.datetime(2017,10,5,10,1)
    date_time = starttime
    run_starttime = datetime.datetime.now()
    
    while(date_time < endtime):
        print("Elapsed time of previous minute {}".format(datetime.datetime.now()-run_starttime))
        run_starttime = datetime.datetime.now()
        print("Processing for timestamp {} current time {}".format(date_time,datetime.datetime.now()))
        filepat    = format(date_time.strftime('%Y%m%d_%H%M'))+"(.*)\.flows"
        filetime    = format(date_time.strftime('%Y:%m:%d:%H:%M'))
        print("filetime >>>>>>>>>>>>>{}<<<<<<<<<<<<<<<<<<<<<<<<<".format(filetime))
        #filenames  = subprocess.check_output(['find', '/mnt/nfs/data2/netflow_filtered_3hrs/', '-name', filepat]).decode().split("\n")[:-1]
        #filenames  = subprocess.check_output(['find', '/mnt/nfs/data2/netflow_filtered_6days', '-name', filepat]).decode().split("\n")[:-1]
        print("Elapsed time for filelist {}".format(datetime.datetime.now()-run_starttime))
        #print("file pattern {} has files: {}".format(filepat, filenames))
        df_min    = pd.DataFrame(columns=df_header.split("|"))
        #flood_ips = ["101.53.130.180", "101.53.130.54", "101.53.130.55", "66.110.0.66"]
        flag = 0
        
        subdf=rawMasterDf[rawMasterDf.apply(ismatched,axis=1)]
        df_min = df_min.append(subdf[subdf.columns[3:]])
        #if filenames:
        #    run_starttime1 = datetime.datetime.now()
        #    for f in filenames:
        #        subdf  = pd.read_csv(f, sep="|")
        #        #subdf.fillna(value=0, inplace=True)
        #        df_min = df_min.append(subdf)
                #df_min = df_min[df_min['IPV4_DST_ADDR'].isin(flood_ips)]
        #    print("Elapsed time for combine files {}".format(datetime.datetime.now()-run_starttime1))
        if not df_min.empty:
            #df_min.fillna(value=0, inplace=True)
            #df_min.iloc[:, [3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]] = df_min.iloc[:, [3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]].apply(pd.to_numeric, errors='coerce')

            run_starttime2= datetime.datetime.now()
            c=0
            for i in range(0,672,25):
                df1 = df_min[(df_min['DEST_CUSTOMER'] >= i) & (df_min['DEST_CUSTOMER']<=(i+25))]
                #df_str = df1.to_json(orient='table')
                df_str = df1.to_csv()
                #print("Length of file pattern {} {}".format(filetime,len(filetime)))
                #print("Length of df_str pattern {}".format(len(df_str)))
                df_str = filetime+df_str
                #date_str = df_str[:len(filetime)]
                #df_str = df_str[len(filetime):]
                #print("Length of date_str pattern {} {}".format(date_str,len(date_str)))
                #print("Length of df_str pattern {}".format(len(df_str)))
                #print("Length of dataframe {}  {} {}".format(i, len(df1), len(df_str)))
                #tasks_to_accomplish.put(df_str)
                queues[c].put(df_str)
                c = c+1
                if (c>=number_of_processes):
                   c = 0
            print("Elapsed time for writing in queue {}".format(datetime.datetime.now()-run_starttime2))

            #if (not firsttime):
            #   firsttime=False
            #   for k in range(0,26):
            #       print(tasks_that_are_done.get())

            #return df_min
            #start_parallel_processing(df_min,date_time)
            #process_minute_minibatch_new(df_min, datetime.datetime.now())
        date_time += datetime.timedelta(minutes=1)

queues = []

def main():
    number_of_task = 10
    global number_of_processes
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []
    global queues

    number_of_processes = 27

    #for i in range(number_of_task):
    #    tasks_to_accomplish.put(i)
    for w in range(number_of_processes):
        q = Queue()
        queues.append(q)
        p = Process(target=do_job, args=(q, tasks_that_are_done))
        processes.append(p)
        #p.start()


    for w in range(number_of_processes):
        processes[w].start()

    df_raw = load_3hr_data(tasks_to_accomplish,tasks_that_are_done)

    #for w in range(number_of_processes):
    #    processes[w].start()


    #while True:
    #    print("in main")
    #    print(tasks_that_are_done.get())
    #    time.sleep(.5)


    # completing process
    for p in processes:
        p.join()

    # print the output
    #while not tasks_that_are_done.empty():
    #    print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()



# In[59]:



def load_3hr_data(tasks_to_accomplish,tasks_that_are_done):
    global queues
    global number_of_processes
    firsttime=True
    starttime = datetime.datetime(2017,9,29,18,1)
    endtime   = datetime.datetime(2017,10,5,10,1)
    date_time = starttime
    run_starttime = datetime.datetime.now()
    while(date_time < endtime):
        print("Elapsed time of previous minute {}".format(datetime.datetime.now()-run_starttime))
        run_starttime = datetime.datetime.now()
        print("Processing for timestamp {} current time {}".format(date_time,datetime.datetime.now()))
        filepat    = format(date_time.strftime('%Y%m%d_%H%M'))+"*.flows"
        filetime    = format(date_time.strftime('%Y:%m:%d:%H:%M'))
        print("filetime >>>>>>>>>>>>>{}<<<<<<<<<<<<<<<<<<<<<<<<<".format(filetime))
        #filenames  = subprocess.check_output(['find', '/mnt/nfs/data2/netflow_filtered_3hrs/', '-name', filepat]).decode().split("\n")[:-1]
        filenames  = subprocess.check_output(['find', '/mnt/nfs/data2/netflow_filtered_6days', '-name', filepat]).decode().split("\n")[:-1]
        print("Elapsed time for filelist {}".format(datetime.datetime.now()-run_starttime))
        #print("file pattern {} has files: {}".format(filepat, filenames))
        df_min    = pd.DataFrame(columns=df_header.split("|"))
        #flood_ips = ["101.53.130.180", "101.53.130.54", "101.53.130.55", "66.110.0.66"]
        flag = 0
        if filenames:
            run_starttime1 = datetime.datetime.now()
            for f in filenames:
                subdf  = pd.read_csv(f, sep="|")
                #subdf.fillna(value=0, inplace=True)
                df_min = df_min.append(subdf)
                #df_min = df_min[df_min['IPV4_DST_ADDR'].isin(flood_ips)]
            print("Elapsed time for combine files {}".format(datetime.datetime.now()-run_starttime1))
        if not df_min.empty:
            #df_min.fillna(value=0, inplace=True)
            #df_min.iloc[:, [3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]] = df_min.iloc[:, [3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]].apply(pd.to_numeric, errors='coerce')

            run_starttime2= datetime.datetime.now()
            c=0
            for i in range(0,672,25):
                df1 = df_min[(df_min['DEST_CUSTOMER'] >= i) & (df_min['DEST_CUSTOMER']<=(i+25))]
                #df_str = df1.to_json(orient='table')
                df_str = df1.to_csv()
                #print("Length of file pattern {} {}".format(filetime,len(filetime)))
                #print("Length of df_str pattern {}".format(len(df_str)))
                df_str = filetime+df_str
                #date_str = df_str[:len(filetime)]
                #df_str = df_str[len(filetime):]
                #print("Length of date_str pattern {} {}".format(date_str,len(date_str)))
                #print("Length of df_str pattern {}".format(len(df_str)))
                #print("Length of dataframe {}  {} {}".format(i, len(df1), len(df_str)))
                #tasks_to_accomplish.put(df_str)
                queues[c].put(df_str)
                c = c+1
                if (c>=number_of_processes):
                   c = 0
            print("Elapsed time for writing in queue {}".format(datetime.datetime.now()-run_starttime2))

            #if (not firsttime):
            #   firsttime=False
            #   for k in range(0,26):
            #       print(tasks_that_are_done.get())

            #return df_min
            #start_parallel_processing(df_min,date_time)
            #process_minute_minibatch_new(df_min, datetime.datetime.now())
        date_time += datetime.timedelta(minutes=1)


# In[ ]:


loaded_model.

