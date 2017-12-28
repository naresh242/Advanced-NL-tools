# -*- coding: utf-8 -*-
"""

This is an example of a simple method that performs bagging

"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# train is the training data
# test is the test data
# y is the target variable  
# bags is number of estimators to run
# seed the random state
# model is an esitimator that incorporates randomness in its fitting function
# return: bagged predictions for the test data
   
def bagging(train , y, test,bags=10,seed=1 ,model=RandomForestRegressor()):
   
   # create array object to hold bagged predictions 
   bagged_prediction=np.zeros(test.shape[0])
   #loop for as many times as we want bags
   for n in range (0, bags):
        model.set_params(random_state=seed + n)# update seed 
        model.fit(train,y) # fit model
        preds=model.predict(test) # predict on test data
        bagged_prediction+=preds # add predictions to bagged predictions 
   #take average of predictions     
   bagged_prediction/= bags   
   return bagged_prediction
