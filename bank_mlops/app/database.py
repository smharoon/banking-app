"""MLOps-owned tables under the mlops schema. Never touches DataOps tables."""
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, Integer, String, Float, DateTime, Text, ForeignKey, UniqueConstraint, JSON, text, insert, select, update, delete
from app.common import COLUMNS, NUMERIC
import pandas as pd

metadata = MetaData(schema='mlops')
files = Table('files', metadata,
    Column('file_id',BigInteger,primary_key=True,autoincrement=True),
    Column('kind',String(12),nullable=False), Column('filename',String(255),nullable=False),
    Column('checksum',String(64),nullable=False), Column('status',String(24),nullable=False),
    Column('row_count',Integer), Column('error',Text),Column('created_at',DateTime(timezone=True)),
    UniqueConstraint('kind','checksum',name='uq_file_kind_checksum'))

training_records=Table('training_records',metadata, Column('id',BigInteger,primary_key=True,autoincrement=True), Column('file_id',BigInteger,ForeignKey('mlops.files.file_id'),nullable=False),Column('source_row',Integer,nullable=False),*[Column(c, Float if c in NUMERIC else String(120)) for c in COLUMNS])
evaluation_records=Table('evaluation_records',metadata, Column('id',BigInteger,primary_key=True,autoincrement=True), Column('file_id',BigInteger,ForeignKey('mlops.files.file_id'),nullable=False),Column('source_row',Integer,nullable=False),*[Column(c, Float if c in NUMERIC else String(120)) for c in COLUMNS])
training_runs=Table('training_runs',metadata,Column('id',BigInteger,primary_key=True,autoincrement=True),Column('file_ids',JSON),Column('algorithm',String(80)),Column('model_version',String(30)),Column('mlflow_run_id',String(80)),Column('metrics',JSON), Column('created_at',DateTime(timezone=True)))
evaluation_metrics=Table('evaluation_metrics',metadata,Column('id',BigInteger,primary_key=True,autoincrement=True),Column('file_id',BigInteger,ForeignKey('mlops.files.file_id')),Column('model_version',String(30)),Column('mlflow_run_id',String(80)),Column('metrics',JSON),Column('created_at',DateTime(timezone=True)),UniqueConstraint('file_id','model_version',name='uq_eval_file_version'))
predictions=Table('predictions',metadata,Column('id',BigInteger,primary_key=True,autoincrement=True),Column('evaluation_id',BigInteger,ForeignKey('mlops.evaluation_metrics.id')),Column('source_row',Integer),Column('actual_y',String(4)),Column('predicted_y',String(4)),Column('probability_yes',Float))

def engine(): return create_engine(os.environ['DATABASE_URL'], pool_pre_ping=True)

def init_db(db):
    with db.begin() as c:
        c.execute(text('CREATE SCHEMA IF NOT EXISTS mlops'))
    metadata.create_all(db)

def insert_file(db, path, kind, df, checksum):
    """Atomic raw record insertion. Existing checksum is never duplicated."""
    rec_table = training_records if kind == 'train' else evaluation_records
    with db.begin() as c:
        existing=c.execute(select(files.c.file_id, files.c.status).where(files.c.kind==kind, files.c.checksum==checksum)).first()
        if existing:
            file_id=int(existing.file_id)
            if existing.status=='completed': return file_id,True
            c.execute(delete(rec_table).where(rec_table.c.file_id==file_id))
            c.execute(update(files).where(files.c.file_id==file_id).values(status='processing',error=None,row_count=len(df),filename=path.name))
        else:
            r=c.execute(insert(files).values(kind=kind, filename=path.name, checksum=checksum,status='processing',row_count=len(df),created_at=datetime.now(timezone.utc)))
            file_id=int(r.inserted_primary_key[0])
        clean=df.astype(object).where(pd.notna(df),None)
        for start in range(0,len(clean),1000):
            rows=[]
            for offset, item in enumerate(clean.iloc[start:start+1000].to_dict('records'),start+1):
                rows.append({**item,'file_id':file_id,'source_row':offset})
            c.execute(insert(rec_table),rows)
    return file_id,False

def mark_file(db,file_id,status,error=None):
    with db.begin() as c: c.execute(update(files).where(files.c.file_id==file_id).values(status=status,error=error))
