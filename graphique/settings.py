import json
from pathlib import Path
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config('.env')
COLUMNS = config('COLUMNS', cast=CommaSeparatedStrings, default=[])
DEBUG = config('DEBUG', cast=bool, default=False)
INDEX = config('INDEX', cast=CommaSeparatedStrings, default=[])
DATASET = {
    'path_or_paths': Path(config('PARQUET_PATH')).resolve(),
    'filters': config('FILTERS', cast=json.loads, default=None),
    'read_dictionary': config('DICTIONARIES', cast=CommaSeparatedStrings, default=None),
    'memory_map': config('MMAP', cast=bool, default=False),
    'use_legacy_dataset': False,
}
