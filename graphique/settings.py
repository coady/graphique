from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config('.env')
COLUMNS = config('COLUMNS', cast=CommaSeparatedStrings, default=None)
DEBUG = config('DEBUG', cast=bool, default=False)
DICTIONARIES = config('DICTIONARIES', cast=CommaSeparatedStrings, default=None)
INDEX = config('INDEX', cast=CommaSeparatedStrings, default=None)
MMAP = config('MMAP', cast=bool, default=True)
PARQUET_PATH = config('PARQUET_PATH')
