"""Ad hoc: combine all pending CSVs, train two models, evaluate 70/30, register champion."""
from pathlib import Path
import pandas as pd
from prefect import flow,task,get_run_logger
from sqlalchemy import insert
from datetime import datetime,timezone
from app.common import ROOT,dirs,load_csv,digest,move_unique
from app.database import engine,init_db,insert_file,mark_file,training_runs
from app.modeling import train_candidates
from app.registry import log_train

@task(retries=2,retry_delay_seconds=5)
def ingest_training(db_url_unused, path_str):
    path=Path(path_str);df=load_csv(path)
    db=engine();init_db(db)
    fid,existing=insert_file(db,path,'train',df,digest(path))
    return fid,existing,df

@flow(name='bank-marketing-train-model',log_prints=True)
def train_model():
    logger=get_run_logger();dirs()
    paths=sorted((ROOT/'train_model/incoming').glob('*.csv'))
    if not paths:
        logger.info('No training CSVs found; nothing to do');return
    db=engine();init_db(db)
    frames=[];file_ids=[];pending=[]
    try:
        for path in paths:
            fid,done,df=ingest_training(None,str(path))
            if done:
                move_unique(path,ROOT/'train_model/processed')
                logger.info('Already trained file: %s',path.name)
                continue
            frames.append(df);file_ids.append(fid);pending.append(path)
        if not frames:
            logger.info('All files were previously processed');return
        combined=pd.concat(frames,ignore_index=True)
        logger.info('Training with %d records from %d CSV files',len(combined),len(frames))
        runs,winner=train_candidates(combined)
        for name,info in runs.items():logger.info('%s metrics: %s',name,info['metrics'])
        version,run_id=log_train(runs,winner,file_ids)
        with db.begin() as c:
            c.execute(insert(training_runs).values(file_ids=file_ids,algorithm=winner,model_version=version,mlflow_run_id=run_id,metrics=runs[winner]['metrics'],created_at=datetime.now(timezone.utc)))
        for fid,path in zip(file_ids,pending):
            mark_file(db,fid,'completed')
            move_unique(path,ROOT/'train_model/processed')
        logger.info('Champion: %s, registered version: %s',winner,version)
    except Exception as exc:
        for fid in file_ids:
            mark_file(db,fid,'failed',str(exc)[:2000])
        logger.exception('Training failed; files remain available in incoming/ for retry')
        raise

if __name__=='__main__':
    import sys
    if '--once' in sys.argv:train_model()
    else:train_model.serve(name='manual-training',pause_on_shutdown=False)
