
# coding: utf-8

# In[1]:


import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion as fu
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin



from matplotlib import pyplot

import pickle
from sklearn.metrics import make_scorer



# In[2]:


class ColumnExtractor(BaseEstimator,TransformerMixin):

    def __init__(self, cols=[1]):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[c])
        if len(col_list)>1:
            return pd.concat(col_list, axis=1)
        else:
            return col_list[0]

    def fit(self, X, y=None):
        return self

    

class Condship(BaseEstimator,TransformerMixin):    

    def __init__(self,cols=[]):
        self.cols=cols
    
        
    def transform(self,X):
        
        col_list = []
        for c in cols:
            col_list.append(X[c])
        if len(col_list)>1:
            return pd.concat(col_list, axis=1)
        else:
            return col_list[0]
    def fit(self, X, y=None):
        return self

    
    
class labelbin(BaseEstimator,TransformerMixin):    

    def __init__(self):
        self.labelbclass=LabelBinarizer(sparse_output=True)
        self.labelbfit=None
    
        
    def transform(self,X):
        return self.labelbfit.transform(X)
        
       
    def fit(self, X, y=None):
        self.labelbfit=self.labelbclass.fit(X)
        return self



# In[3]:


train=pd.DataFrame()
train["X"]=["book","b","c","king","king","b","king","king","king","set"]
train["Y"]=["book","b","c","king","king","b","king","king","king","set"]
train["W"]=[1,2,3,4,5,6,7,8,9,10]
train["Z"]=[1,2,3,4,5,6,7,8,9,10]


# In[15]:


headest=[]
fuest=[]

###########namep

namep_est=[]
selectname=ColumnExtractor(cols=["X"])
CountVectorizer_name=CountVectorizer(min_df=3)
namep_est.append(("selectname",selectname))
namep_est.append(("CountVectorizer_name",CountVectorizer_name))
namep=Pipeline(namep_est)
fuest.append(("namep",namep))

#############catp
headest=[]
fuest=[]

###########namep

namep_est=[]
selectname=ColumnExtractor(cols=["X"])
CountVectorizer_name=CountVectorizer(min_df=3)
namep_est.append(("selectname",selectname))
namep_est.append(("CountVectorizer_name",CountVectorizer_name))
namep=Pipeline(namep_est)
fuest.append(("namep",namep))

#############catp

catp_est=[]
catid=ColumnExtractor(cols=["X"])
CountVectorizer_cat=CountVectorizer(min_df=3)
catp_est.append(("catid",catid))
catp_est.append(("CountVectorizer_cat",CountVectorizer_cat))
catp=Pipeline(catp_est)
fuest.append(("catp",catp))

##########desc

descp_est=[]
desc=ColumnExtractor(cols=["X"])
Tfidf_desc=TfidfVectorizer(max_features=4,ngram_range=(1, 3),stop_words='english')
descp_est.append(("desc",desc))
descp_est.append(("Tfidf_desc",Tfidf_desc))
descp=Pipeline(descp_est)
fuest.append(("descp",descp))

#############brandp

brandp_est=[]
brand=ColumnExtractor(cols=["X"])
labelbin_brand=labelbin()
brandp_est.append(("brand",brand))
brandp_est.append(("labelbin_brand",labelbin_brand))
brandp=Pipeline(brandp_est)
fuest.append(("brandp",brandp))

###########ship & condition

condsupp_est=[]
con_ship_col=ColumnExtractor(cols=["X"])

condsupp_est.append(("con_ship_col",con_ship_col))
condsupp_est.append(("labelbin_brand",labelbin_brand))
condsupp=Pipeline(condsupp_est)
fuest.append(("condsupp",condsupp))



#feature union obj
feature=fu(fuest,n_jobs=1)

#final headpipe
headest.append(("feature",feature))
model=lgb.LGBMRegressor(n_jobs=-1)
headest.append(("model",model))
headpipe=Pipeline(headest)


headpipe.fit(train,train["Z"])
catp_est=[]
catid=ColumnExtractor(cols=["X"])
CountVectorizer_cat=CountVectorizer(min_df=3)
catp_est.append(("catid",catid))
catp_est.append(("CountVectorizer_cat",CountVectorizer_cat))
catp=Pipeline(catp_est)
fuest.append(("catp",catp))

##########desc

descp_est=[]
desc=ColumnExtractor(cols=["X"])
Tfidf_desc=TfidfVectorizer(max_features=4,ngram_range=(1, 3),stop_words='english')
descp_est.append(("desc",desc))
descp_est.append(("Tfidf_desc",Tfidf_desc))
descp=Pipeline(descp_est)
fuest.append(("descp",descp))

#############brandp

brandp_est=[]
brand=ColumnExtractor(cols=["X"])
labelbin_brand=labelbin()
brandp_est.append(("brand",brand))
brandp_est.append(("labelbin_brand",labelbin_brand))
brandp=Pipeline(brandp_est)
fuest.append(("brandp",brandp))

###########ship & condition

condsupp_est=[]
con_ship_col=ColumnExtractor(cols=["X"])

condsupp_est.append(("con_ship_col",con_ship_col))
condsupp_est.append(("labelbin_brand",labelbin_brand))
condsupp=Pipeline(condsupp_est)
fuest.append(("condsupp",condsupp))



#feature union obj
feature=fu(fuest,n_jobs=1)

#final headpipe
headest.append(("feature",feature))
model=lgb.LGBMRegressor(n_jobs=-1)
headest.append(("model",model))
headpipe=Pipeline(headest)


headpipe.fit(train,train["Z"])


# In[4]:





pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=["X"])),
                                                 ("CountVectorizer_cat",CountVectorizer(min_df=3))
                                                ]))]))])

print("done")
pipe.fit(train)


# In[ ]:


model=lgb.LGBMRegressor(n_jobs=-1)
pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",CountVectorizer(min_df=3))])),
                              ("catp",Pipeline([("catid",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",CountVectorizer())])),
                             ("descp",Pipeline([("desc",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",TfidfVectorizer(max_features=4,ngram_range=(1, 3),stop_words='english'))])),
                             ("brandp",Pipeline([("brand",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",labelbin())])),
                             ("condsupp",Pipeline([("ship",ColumnExtractor(cols=['W','W']))]))],n_jobs=-1)),
              ("model",model)])

print("done")
pipe.fit(train)


# In[3]:


pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=[0]))])),
                              ("catp",Pipeline([("catid",ColumnExtractor(cols=[1]))]))]))])


# In[7]:


lb=LabelBinarizer(sparse_output=True)
lb.fit(train["X"])
lb.transform(train["X"])



# In[ ]:


f=fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",CountVectorizer())]))
                              ])

f.fit(train)
f.transform(train)


# In[13]:



pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",CountVectorizer())]))
                              ]))])



# In[29]:



pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",CountVectorizer())])),
                              ("catp",Pipeline([("catid",ColumnExtractor(cols=["Y"])),("CountVectorizer_cat",CountVectorizer())])),
                             ("descp",Pipeline([("desc",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",TfidfVectorizer(max_features=4,ngram_range=(1, 3),stop_words='english'))])),
                             ]))])





# In[4]:



pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",CountVectorizer())])),
                              ("catp",Pipeline([("catid",ColumnExtractor(cols=["Y"])),("CountVectorizer_cat",CountVectorizer())])),
                             ("descp",Pipeline([("desc",ColumnExtractor(cols=["X"])),("CountVectorizer_cat",TfidfVectorizer(max_features=4,ngram_range=(1, 3),stop_words='english'))])),
                             ("brandp",Pipeline([("brand",ColumnExtractor(cols=["Y"])),("CountVectorizer_cat",labelbin())])),
                             ("condsupp",Pipeline([("ship",ColumnExtractor(cols=["X","Y"]))]))],n_jobs=-1))])





# In[6]:


pipe=Pipeline([("feature",fu([("namep",Pipeline([("selectname",ColumnExtractor(cols=[1])),("CountVectorizer_name",CountVectorizer(min_df=3))])),
                              ("catp",Pipeline([("catid",ColumnExtractor(cols=[2])),("CountVectorizer_cat",CountVectorizer())])),
                              ("descp",Pipeline([("desc",ColumnExtractor(cols=[7])),("CountVectorizer_cat",TfidfVectorizer(max_features=4,ngram_range=(1, 3),stop_words='english'))])),
                              ("brandp",Pipeline([("brand",ColumnExtractor(cols=[4])),("CountVectorizer_cat",LabelBinarizer(sparse_output=True))])),
                              ("condsupp",Pipeline([("brand",ColumnExtractor(cols=[2,6]))]))],n_jobs=-1))])


# In[ ]:


pipe.fit(train) 


# In[20]:


train=pd.DataFrame()
train["X"]=[1,2,3,4,5,6,7,8,9,10]
train["Y"]=[1,2,3,4,5,6,7,8,9,10]

pp=pipeline([("feature",fu([
    ('ss', ss()), # can pass in either a pipeline
    ('mm', mm()) # or a transformer
],n_jobs=-1)),("Ridge",Ridge())])


pp.fit(train[["X"]],train["Y"])

pp.predict(train[["X"]])

