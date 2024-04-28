"""
GraphQL scalars.
"""

import functools
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Union, no_type_check
import pyarrow as pa
import strawberry


def parse_long(value) -> int:
    if isinstance(value, int):
        return value
    raise TypeError(f"Long cannot represent value: {value}")


@no_type_check
def parse_duration(value: str):
    months = days = seconds = 0
    d_val, _, t_val = value.partition('T')
    parts = re.split(r'(-?\d+\.?\d*)', d_val.lower() + t_val)
    if parts.pop(0) != 'p':
        raise ValueError("Duration format must start with `P`")
    for num, key in zip(parts[::2], parts[1::2]):
        if n := float(num) if '.' in num else int(num):
            if key in 'ym':
                months += n * 12 if key == 'y' else n
            elif key in 'wd':
                days += n * 7 if key == 'w' else n
            elif key in 'HMS':
                seconds += n * {'H': 3600, 'M': 60, 'S': 1}[key]
            else:
                raise ValueError(f"Invalid duration field: {key.upper()}")
    if months:
        return pa.MonthDayNano([months, days, int(seconds * 1_000_000_000)])
    return timedelta(days, seconds)


@functools.singledispatch
def duration_isoformat(td: timedelta) -> str:
    days = f'{td.days}D' if td.days else ''
    fraction = f'.{td.microseconds:06}' if td.microseconds else ''
    return f'P{days}T{td.seconds}{fraction}S'


@duration_isoformat.register
def _(mdn: pa.MonthDayNano) -> str:
    months = f'{mdn.months}M' if mdn.months else ''
    days = f'{mdn.days}D' if mdn.days else ''
    seconds, nanoseconds = divmod(mdn.nanoseconds, 1_000_000_000)
    fraction = f'.{nanoseconds:09}' if nanoseconds else ''
    return f'P{months}{days}T{seconds}{fraction}S'


Long = strawberry.scalar(int, name='Long', description="64-bit int", parse_value=parse_long)
Duration = strawberry.scalar(
    Union[timedelta, pa.MonthDayNano],
    name='Duration',
    description="Duration (isoformat)",
    specified_by_url="https://en.wikipedia.org/wiki/ISO_8601#Durations",
    serialize=duration_isoformat,
    parse_value=parse_duration,
)
scalar_map = {
    bytes: strawberry.scalars.Base64,
    dict: strawberry.scalars.JSON,
    timedelta: Duration,
    pa.MonthDayNano: Duration,
}

type_map = {
    pa.lib.Type_BOOL: bool,
    pa.lib.Type_UINT8: int,
    pa.lib.Type_INT8: int,
    pa.lib.Type_UINT16: int,
    pa.lib.Type_INT16: int,
    pa.lib.Type_UINT32: Long,
    pa.lib.Type_INT32: int,
    pa.lib.Type_UINT64: Long,
    pa.lib.Type_INT64: Long,
    pa.lib.Type_HALF_FLOAT: float,
    pa.lib.Type_FLOAT: float,
    pa.lib.Type_DOUBLE: float,
    pa.lib.Type_DECIMAL128: Decimal,
    pa.lib.Type_DECIMAL256: Decimal,
    pa.lib.Type_DATE32: date,
    pa.lib.Type_DATE64: date,
    pa.lib.Type_TIMESTAMP: datetime,
    pa.lib.Type_TIME32: time,
    pa.lib.Type_TIME64: time,
    pa.lib.Type_DURATION: timedelta,
    pa.lib.Type_INTERVAL_MONTH_DAY_NANO: pa.MonthDayNano,
    pa.lib.Type_BINARY: bytes,
    pa.lib.Type_FIXED_SIZE_BINARY: bytes,
    pa.lib.Type_LARGE_BINARY: bytes,
    pa.lib.Type_STRING: str,
    pa.lib.Type_LARGE_STRING: str,
    pa.lib.Type_LIST: list,
    pa.lib.Type_FIXED_SIZE_LIST: list,
    pa.lib.Type_LARGE_LIST: list,
    pa.lib.Type_STRUCT: dict,
}
