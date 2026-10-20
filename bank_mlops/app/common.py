"""Shared CSV contract and configuration. Independent of DataOps tables."""
import hashlib
import os
from pathlib import Path
import pandas as pd

COLUMNS = ['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']
NUMERIC = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
FEATURES = [c for c in COLUMNS if c not in ('duration','y')]
CATS = [c for c in FEATURES if c not in NUMERIC]
NUM_FEATURES = [c for c in FEATURES if c in NUMERIC]
ROOT = Path(os.getenv('MLOPS_DATA_DIR', '/app')).resolve()
MODEL_NAME = os.getenv('MODEL_NAME','bank-marketing-best-model')

def load_csv(path: Path, require_target=True):
    data = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    data.columns = data.columns.str.strip()
    required = COLUMNS if require_target else FEATURES
    missing = sorted(set(required) - set(data.columns))
    if missing: raise ValueError(f'{path.name}: missing columns {missing}')
    if data.empty: raise ValueError(f'{path.name}: CSV is empty')
    data = data[COLUMNS if require_target else FEATURES].copy()
    for col in NUMERIC:
        if col in data: data[col] = pd.to_numeric(data[col], errors='raise')
    if require_target and (data['y'].isna().any() or not data['y'].isin(['yes','no']).all()):
        raise ValueError(f'{path.name}: y must be yes or no')
    return data

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def dirs():
    for sub in ('train_model/incoming','train_model/processed','train_model/failed','evaluate_model/incoming','evaluate_model/processed','evaluate_model/failed','outputs'):
        (ROOT/sub).mkdir(parents=True, exist_ok=True)

def move_unique(src, target_dir):
    target_dir.mkdir(parents=True,exist_ok=True)
    dest=target_dir/src.name
    if dest.exists():
        import uuid
        dest=target_dir/f'{src.stem}_{uuid.uuid4().hex[:8]}{src.suffix}'
    return src.rename(dest)
