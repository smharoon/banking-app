"""Scheduled: each labeled CSV evaluates saved champion; no model retraining."""
from pathlib import Path
from datetime import datetime,timezone
from sqlalchemy import select,insert
from prefect import flow,task,get_run_logger
from app.common import ROOT,dirs,load_csv,digest,move_unique
from app.database import engine,init_db,insert_file,mark_file,evaluation_metrics,predictions
from app.modeling import infer,scores
from app.registry import champion,log_eval

@task(retries=2,retry_delay_seconds=5)
def evaluate_one(path_str):
    path=Path(path_str);db=engine();init_db(db)
    frame=load_csv(path);file_id,duplicate=insert_file(db,path,'evaluate',frame,digest(path))
    try:
        model,version=champion()
    except Exception as exc:
        mark_file(db,file_id,'failed',str(exc)[:2000])
        raise
    with db.connect() as c:
        existing=c.execute(select(evaluation_metrics.c.id).where(evaluation_metrics.c.file_id==file_id,evaluation_metrics.c.model_version==version)).first()
    if existing:
        mark_file(db,file_id,'completed')
        move_unique(path,ROOT/'evaluate_model/processed')
        return {'file':path.name,'status':'already evaluated','model_version':version}
    try:
        prediction,proba=infer(model,frame)
        metrics=scores(frame['y'],prediction,proba)
        run_id=log_eval(metrics,file_id,version)
        with db.begin() as c:
            result=c.execute(insert(evaluation_metrics).values(file_id=file_id,model_version=version,mlflow_run_id=run_id,metrics=metrics,created_at=datetime.now(timezone.utc)))
            eval_id=int(result.inserted_primary_key[0])
            payload=[{'evaluation_id':eval_id,'source_row':i,'actual_y':str(actual),'predicted_y':str(pred),'probability_yes':float(prob)} for i,(actual,pred,prob) in enumerate(zip(frame['y'],prediction,proba),1)]
            for start in range(0,len(payload),1000):c.execute(insert(predictions),payload[start:start+1000])
        output=ROOT/'outputs'/f'evaluation_{file_id}_v{version}.csv'
        result_frame=frame.copy();result_frame['predicted_y']=prediction;result_frame['probability_yes']=proba
        result_frame.to_csv(output,index=False)
        mark_file(db,file_id,'completed')
        move_unique(path,ROOT/'evaluate_model/processed')
        return {'file':path.name,'status':'completed','model_version':version,'metrics':metrics,'output':str(output)}
    except Exception as exc:
        mark_file(db,file_id,'failed',str(exc)[:2000]);raise

@flow(name='bank-marketing-evaluate-model',log_prints=True)
def evaluate_model():
    dirs();log=get_run_logger()
    paths=sorted((ROOT/'evaluate_model/incoming').glob('*.csv'))
    if not paths:
        log.info('No evaluation CSVs found');return
    failures=[]
    for path in paths:
        try:log.info('Result: %s',evaluate_one(str(path)))
        except Exception as exc:
            failures.append(path.name)
            log.error('Evaluation failed for %s: %s. File stays in incoming for retry.',path.name,exc)
    if failures:raise RuntimeError('Failed evaluation files: '+', '.join(failures))

if __name__=='__main__':
    import sys
    if '--once' in sys.argv:evaluate_model()
    else:evaluate_model.serve(name='every-2-minutes',interval=120,pause_on_shutdown=False,limit=1)
