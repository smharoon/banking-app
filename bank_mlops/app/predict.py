"""CLI: python -m app.predict --input customers.csv --output predictions.csv"""
import argparse
from pathlib import Path
from app.common import load_csv
from app.modeling import infer
from app.registry import champion

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    df=load_csv(Path(args.input),require_target=False)
    model,version=champion()
    prediction,proba=infer(model,df)
    out=df.copy();out['predicted_y']=prediction;out['probability_yes']=proba
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest,index=False)
    print(f'Created {dest} with {len(out)} predictions using champion version {version}')
if __name__=='__main__':main()
