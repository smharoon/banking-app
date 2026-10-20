"""Train two pipelines and evaluate an immutable registered champion version."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from app.common import FEATURES, NUM_FEATURES, CATS

def make_pipeline(estimator):
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler', StandardScaler())])
    categoric=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='unknown')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    prep=ColumnTransformer([('numeric',numeric,NUM_FEATURES),('categorical',categoric,CATS)])
    return Pipeline([('prep',prep),('classifier',estimator)])

def scores(actual, predicted, prob):
    y=np.asarray(actual)=='yes'; yp=np.asarray(predicted)=='yes'
    result={'accuracy':float(accuracy_score(y,yp)),'precision':float(precision_score(y,yp,zero_division=0)), 'recall':float(recall_score(y,yp,zero_division=0)), 'f1':float(f1_score(y,yp,zero_division=0))}
    if len(np.unique(y))==2: result['roc_auc']=float(roc_auc_score(y,prob))
    result['tn'],result['fp'],result['fn'],result['tp']=map(int,confusion_matrix(y,yp,labels=[False,True]).ravel())
    return result

def infer(model, df):
    X=df[FEATURES].copy()
    pred=model.predict(X)
    proba=model.predict_proba(X)
    positive=list(model.classes_).index('yes')
    return pred,proba[:,positive]

def train_candidates(df):
    if df['y'].nunique()!=2 or df['y'].value_counts().min()<2: raise ValueError('Training requires at least two records of each y class')
    X=df[FEATURES]; y=df['y']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,random_state=42,stratify=y)
    candidates={'LogisticRegression':LogisticRegression(max_iter=1500,class_weight='balanced'), 'RandomForest':RandomForestClassifier(n_estimators=120,max_depth=12,random_state=42,n_jobs=-1,class_weight='balanced_subsample')}
    runs={}
    for name,estimator in candidates.items():
        model=make_pipeline(estimator)
        model.fit(Xtr,ytr)
        predictions,prob=infer(model,Xte)
        runs[name]={'model':model,'metrics':scores(yte,predictions,prob)}
    winner=max(runs,key=lambda name:(runs[name]['metrics']['f1'],runs[name]['metrics'].get('roc_auc',-1)))
    return runs,winner
