from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=50, base_estimator=dtree,learning_rate=.5)
#n_estimators=100,learning_rate=1 for few models these values worked for me
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight 
clf.fit(x_train,y_train)
