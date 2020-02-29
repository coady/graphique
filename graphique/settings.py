from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config('.env')
DEBUG = config('DEBUG', cast=bool, default=False)
INDEX = config('INDEX', cast=CommaSeparatedStrings, default='')
MMAP = config('MMAP', cast=bool, default=True)
PARQUET_PATH = config('PARQUET_PATH')
