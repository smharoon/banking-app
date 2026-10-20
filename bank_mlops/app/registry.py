"""MLflow tracking and champion alias management."""
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from app.common import MODEL_NAME

def configure(): mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

def log_train(runs,winner,file_ids):
    configure(); mlflow.set_experiment('bank-marketing-training')
    winning_run=None
    for name,info in runs.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({'mode':'train','algorithm':name,'source_file_ids':','.join(map(str,file_ids))})
            mlflow.log_metrics(info['metrics'])
            mlflow.log_param('train_test_split','70/30')
            mlflow.log_param('exclude_duration',True)
            if name==winner:
                mlflow.sklearn.log_model(info['model'],name='model',registered_model_name=MODEL_NAME)
                winning_run=run.info.run_id
    client=MlflowClient()
    versions=client.search_model_versions(f"name='{MODEL_NAME}'")
    chosen=[v for v in versions if v.run_id==winning_run]
    if not chosen: raise RuntimeError('Winning model registry version not found')
    version=max(chosen,key=lambda v:int(v.version))
    client.set_registered_model_alias(MODEL_NAME,'champion',version.version)
    return str(version.version),winning_run

def champion():
    configure()
    client=MlflowClient()
    version=client.get_model_version_by_alias(MODEL_NAME,'champion')
    return mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{version.version}'),str(version.version)

def log_eval(metrics,file_id,version):
    configure();mlflow.set_experiment('bank-marketing-evaluation')
    with mlflow.start_run(run_name=f'file-{file_id}-v{version}') as run:
        mlflow.set_tags({'mode':'evaluate','file_id':str(file_id),'model_version':version,'registered_model':MODEL_NAME})
        mlflow.log_metrics(metrics)
        return run.info.run_id
