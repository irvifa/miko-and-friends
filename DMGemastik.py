'''
Created on Aug 13, 2016

@author: ASUS
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import math


dat = pd.read_csv("separate.csv")
dat = dat.fillna(0)
tmp = dat["Expected"]
tmp = np.sign(tmp)
dat["Expected"] = tmp
dat.to_csv("hehe.csv", index=False)
Y = dat["Expected"]
dat = dat.drop("Expected", 1)
dat = dat.drop("Id",1)
dat = dat.drop("Kdp", 1)
radar_quality = dat[dat["RadarQualityIndex"] < 999]
radar_quality = radar_quality[radar_quality["RadarQualityIndex"] > 0]
print radar_quality.mean()
k = dat["RadarQualityIndex"].copy()
k[k < 0] = 0.0
k[k > 1] = 0.321765
dat["RadarQualityIndex"] = k
print dat
# print radar_quality["RadarQualityIndex"].mean()
# dat = dat.drop("HybridScan", 1)
X = dat.copy()
 
print X
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
print clf.feature_importances_ 

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                     test_size=0.2,
#                                                     random_state=1)
# logistic = linear_model.LogisticRegression()
# rbm = BernoulliRBM(random_state=0, verbose=False)
# 
# classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
# 
# rbm.learning_rate = 0.06
# rbm.n_iter = 20
# # More components tend to give better prediction performance, but larger
# # fitting time
# rbm.n_components = 100
# logistic.C = 1000.0
# 
# # Training RBM-Logistic Pipeline
# print "X_Train"
# print X_train
# print "Y_Train"
# print Y_train
# classifier.fit(X_train, Y_train)
# 
# # Training Logistic regression
# logistic_classifier = linear_model.LogisticRegression(C=100.0)
# logistic_classifier.fit(X_train, Y_train)
# 
# print()
# print("Logistic regression using RBM features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         classifier.predict(X_test))))
# 
# print("Logistic regression using raw pixel features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,logistic_classifier.predict(X_test))))
# 
# joblib.dump(logistic_classifier, 'logistic.pkl')
# joblib.dump(classifier,"rbm.pkl")




