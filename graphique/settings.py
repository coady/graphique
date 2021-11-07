import json
from pathlib import Path
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config('.env')
PARQUET_PATH = Path(config('PARQUET_PATH')).resolve()
INDEX = config('INDEX', cast=CommaSeparatedStrings, default=[])
FEDERATED = config('FEDERATED', default='')
DEBUG = config('DEBUG', cast=bool, default=False)
DICTIONARIES = config('DICTIONARIES', cast=CommaSeparatedStrings, default=[])
COLUMNS = config('COLUMNS', cast=CommaSeparatedStrings, default=[])
FILTERS = config('FILTERS', cast=json.loads, default='{}')
