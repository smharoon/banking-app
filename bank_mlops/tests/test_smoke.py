import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.common import COLUMNS, load_csv, digest
from app.modeling import train_candidates, infer, scores

def sample_frame(n=80):
    r=[]
    for i in range(n):
        row=dict.fromkeys(COLUMNS)
        row.update(age=20+i%50,job=['technician','admin.'][i%2],marital='married',education='basic.4y',default='no',housing='yes',loan='no',contact='cellular',month='may',day_of_week='mon',duration=200,campaign=1,pdays=999,previous=0,poutcome='nonexistent',**{'emp.var.rate':1.1,'cons.price.idx':93.994,'cons.conf.idx':-36.4,'euribor3m':4.857,'nr.employed':5191,'y':'yes' if i%4==0 else 'no'})
        r.append(row)
    return pd.DataFrame(r)[COLUMNS]

def test_train_and_evaluate_models():
    df=sample_frame()
    runs,winner=train_candidates(df)
    assert set(runs)=={'LogisticRegression','RandomForest'}
    assert 0<=runs[winner]['metrics']['f1']<=1
    predicted,prob=infer(runs[winner]['model'],df)
    assert len(predicted)==len(df)
    assert 0<=scores(df['y'],predicted,prob)['accuracy']<=1

def test_csv_validation(tmp_path):
    path=tmp_path/'bank.csv';sample_frame(2).to_csv(path,sep=';',index=False)
    assert load_csv(path).shape==(2,21)
    assert len(digest(path))==64
