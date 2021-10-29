import json
from pathlib import Path
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config('.env')
COLUMNS = config('COLUMNS', cast=CommaSeparatedStrings, default=[])
DEBUG = config('DEBUG', cast=bool, default=False)
INDEX = config('INDEX', cast=CommaSeparatedStrings, default=[])
PARQUET_PATH = Path(config('PARQUET_PATH')).resolve()
FILTERS = config('FILTERS', cast=json.loads, default='{}')
DICTIONARIES = config('DICTIONARIES', cast=CommaSeparatedStrings, default=[])
