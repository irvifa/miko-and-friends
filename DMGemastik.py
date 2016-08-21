'''
Created on Aug 13, 2016

@author: ASUS
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.svm.classes import LinearSVC

def remap(x):
    if x == np.float64(-99900):
        return -14
    elif x == np.float64(-99901):
        return np.NaN
    elif x == np.float64(-99903):
        return np.NaN
    elif x == np.float64(999):
        return np.NaN
    else:
        return x
        
class DeepLearning:
    
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.Y = self.data["Expected"]
        self.classifier = None
    def preprocessing(self):
        k = self.data["Expected"]
        k[ k > 0 ] = 1
        self.data["Expected"] = k
        self.data = self.data.applymap(lambda x: remap(x))
        data = self.data.drop("Expected", 1)
        data = data.drop("Id",1)
        data = data.drop("HydrometeorType",1)
        for column in data:
            dat = data[column]
            mi = dat.min(axis=1,skipna=True)
            ma = dat.max(axis=1,skipna=True)
            dat = map(lambda x: (x - mi)/(ma-mi), dat)
            self.data[column] = dat
        print self.data  
    
    def after_preprocessing(self):
        dat = self.data.drop(self.data.columns[0],1)
        dat = dat.drop(self.data.columns[1],1)
        dat = dat.drop(self.data.columns[6],1)
        dat = dat.drop(self.data.columns[7],1)
        dat = dat.drop('IsNoEcho1',1)
        dat = dat.drop('IsIceCrystal',1)
        dat = dat.drop('IsGraupel2',1)
        dat = dat.drop("Expected",1)
        self.data = dat
        self.data = self.data.fillna(0)
#         self.data.to_csv("train_preprocessed.csv",index=False)
        
        
    def get_feature_importance(self):
        clf = ExtraTreesClassifier()
        clf = clf.fit(self.data,self.Y)
        k = self.data.columns
        l = zip(clf.feature_importances_,k)
        print sorted(l,reverse=True)
    
    def  train(self):
        rbm = BernoulliRBM(random_state=0, verbose=False)
        svc = LinearSVC(C=1000.0,class_weight='balanced',max_iter=100)
        classifier = Pipeline(steps=[('rbm', rbm), ('svm', svc)])
        
        rbm.learning_rate = 0.05
        rbm.n_iter = 30
        # More components tend to give better prediction performance, but larger
        # fitting time
        rbm.n_components = 150
        
        classifier.fit(self.data, self.Y)
        self.classifier = classifier
        joblib.dump(classifier,"rbm.pkl")
        
    def load(self):
        self.classifier = joblib.load("rbm.pkl")
        
    def validation(self):
        val_set = pd.read_csv("data_validation.csv") 
        dat = val_set.drop(self.data.columns[0],1)
        dat = dat.drop(val_set.columns[1],1)
        dat = dat.drop(val_set.columns[6],1)
        dat = dat.drop(val_set.columns[7],1)
        Y = val_set["Expected"]
        dat = dat.drop("Expected",1)
        dat = dat.fillna(0)
        print metrics.classification_report(Y,self.classifier.predict(dat))
        
dl = DeepLearning("data_train.csv")
dl.after_preprocessing()
dl.train()
dl.validation()


