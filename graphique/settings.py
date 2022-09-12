import json
from pathlib import Path
from starlette.config import Config

config = Config('.env')
PARQUET_PATH = Path(config('PARQUET_PATH')).resolve()
FEDERATED = config('FEDERATED', default='')
DEBUG = config('DEBUG', cast=bool, default=False)
COLUMNS = config('COLUMNS', cast=json.loads, default=None)
FILTERS = config('FILTERS', cast=json.loads, default=None)
