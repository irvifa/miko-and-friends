'''
Created on Aug 13, 2016

@author: ASUS
'''
import numpy as np
import pandas as pd
from sklearn import metrics, linear_model, grid_search
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.svm.classes import LinearSVC


def remap(x):
#     if x == np.float64(-99900):
#         return -14
#     elif x == np.float64(-99901):
#         return np.NaN
#     elif x == np.float64(-99903):
#         return np.NaN
#     elif x == np.float64(999):
#         return np.NaN
#     else:
        np.reshape()  # @UndefinedVariable
        return x
        
class DeepLearning:
    
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.Y = self.data["Expected"]
        self.X = None
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
        
        
        dat = dat.drop('IswetSnow',1)
        dat = dat.drop('IsDrySnow',1)
        dat = dat.drop('IsBigDrops',1)
        dat = dat.drop('IsModerateRain2',1)
        dat = dat.drop('IsHeavyRain',1)
        self.data = dat
        imp = Imputer(strategy='mean')
        imp.fit(self.data)
        self.X = imp.transform(self.data)
#         self.data.to_csv("train_preprocessed.csv",index=False)
        
        
    def get_feature_importance(self):
        clf = ExtraTreesClassifier()
        clf = clf.fit(self.X,self.Y)
        k = self.data.columns
        l = zip(clf.feature_importances_,k)
        print sorted(l,reverse=True)
    
    def  train_with_svm(self):
        rbm = BernoulliRBM(random_state=0, verbose=False)
        svc = LinearSVC(C=10.0,class_weight='balanced',max_iter=100)
        classifier = Pipeline(steps=[('rbm', rbm), ('svm', svc)])
        
        rbm.learning_rate = 0.05
        rbm.n_iter = 30
        # More components tend to give better prediction performance, but larger
        # fitting time
        rbm.n_components = 100
        
        classifier.fit(self.X, self.Y)
        self.classifier = classifier
        joblib.dump(classifier,"rbm.pkl")
    
    def train_with_logistic(self):
        rbm = BernoulliRBM(random_state=0, verbose=False)
        logistic = linear_model.LogisticRegression(C=100)
        classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
        
        rbm.learning_rate = 0.05
        rbm.n_iter = 30
        # More components tend to give better prediction performance, but larger
        # fitting time
        rbm.n_components = 30
        
        classifier.fit(self.X, self.Y)
        self.classifier = classifier
        joblib.dump(classifier,"rbm-logistic.pkl")
        
    def train_deep_boltzman(self):
        rbm1 = BernoulliRBM(random_state=0, verbose=False)
        logistic = linear_model.LogisticRegression(class_weight='balanced')

        classifier = Pipeline(steps=[('rbm', rbm1), ('logistic', logistic)])
        # More components tend to give better prediction performance, but larger
        # fitting time
        
        params = {
        "rbm__learning_rate": [0.1, 0.03, 0.01],
        "rbm__n_iter": [20, 40, 80],
        "rbm__n_components": [50, 75, 100],
        "logistic__C": [1.0, 10.0, 100.0]}
#         gs = grid_search.GridSearchCV(classifier,params)
#         gs.fit(self.X, self.Y)
        print "grid search done, training pipelined classifier"
        rbm1.n_components = 100
        rbm1.n_iter = 40
        rbm1.learning_rate = 0.01
        logistic.C = 10.0
        classifier.fit(self.X, self.Y)
        self.classifier = classifier
        "classification"
        joblib.dump(classifier,"two-layerRbm-logistic.pkl")
        
    def train_logistic(self):
        logistic = linear_model.LogisticRegression(C=10, class_weight="balanced")
        logistic.fit(self.data, self.Y)
        self.classifier = logistic
        joblib.dump(logistic,"logistic.pkl")
        
    def load(self):
        self.classifier = joblib.load("rbm-logistic.pkl")
        
    def validation(self):
        val_set = pd.read_csv("data_validation.csv")
        Y = val_set["Expected"]
        c = val_set.columns
        val_set = val_set.drop(c[0],1)
        val_set = val_set.drop(c[1],1)
        val_set = val_set.drop(c[6],1)
        val_set = val_set.drop(c[7],1)
        val_set = val_set.drop("Expected",1)
        val_set = val_set.drop('IsNoEcho1',1)
        val_set = val_set.drop('IsIceCrystal',1)
        val_set = val_set.drop('IsGraupel2',1)
        val_set = val_set.drop('IswetSnow',1)
        val_set = val_set.drop('IsDrySnow',1)
        val_set = val_set.drop('IsBigDrops',1)
        val_set = val_set.drop('IsModerateRain2',1)
        val_set = val_set.drop('IsHeavyRain',1)
        imp = Imputer(strategy='mean')
        imp.fit(val_set)
        X = imp.transform(val_set)
        print metrics.classification_report(Y,self.classifier.predict(X))
        

dl = DeepLearning("data_train.csv")
dl.after_preprocessing()
dl.train_deep_boltzman()
dl.validation()


