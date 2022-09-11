import json
from pathlib import Path
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config('.env')
PARQUET_PATH = Path(config('PARQUET_PATH')).resolve()
FEDERATED = config('FEDERATED', default='')
DEBUG = config('DEBUG', cast=bool, default=False)
DICTIONARIES = config('DICTIONARIES', cast=CommaSeparatedStrings, default=[])
COLUMNS = config('COLUMNS', cast=json.loads, default=None)
FILTERS = config('FILTERS', cast=json.loads, default=None)
