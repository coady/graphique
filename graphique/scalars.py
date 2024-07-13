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
    multipliers = {'y': 12, 'w': 7, 'H': 3600, 'M': 60}
    for num, key in zip(parts[::2], parts[1::2]):
        value = (float if '.' in num else int)(num) * multipliers.get(key, 1)
        if key in 'ym':
            months += value
        elif key in 'wd':
            days += value
        elif key in 'HMS':
            seconds += value
        else:
            raise ValueError(f"Invalid duration field: {key.upper()}")
    if set(d_val).isdisjoint('YM'):
        return timedelta(days, seconds)
    return pa.MonthDayNano([months, days, int(seconds * 1_000_000_000)])


@functools.singledispatch
def duration_isoformat(months: int, days: int, seconds: int, fraction: str = '.') -> str:
    minutes, seconds = divmod(seconds, 60)
    items = zip('YMDHM', divmod(months, 12) + (days,) + divmod(minutes, 60))
    year, month, day, hour, minute = (f'{value}{key}' if value else '' for key, value in items)
    fraction = fraction.rstrip('0').rstrip('.')
    return f'P{year}{month}{day}T{hour}{minute}{seconds}{fraction}S'


@duration_isoformat.register
def _(td: timedelta) -> str:  # type: ignore
    return duration_isoformat(0, td.days, td.seconds, f'.{td.microseconds:06}')


@duration_isoformat.register
def _(mdn: pa.MonthDayNano) -> str:
    seconds, nanoseconds = divmod(mdn.nanoseconds, 1_000_000_000)
    value = duration_isoformat(mdn.months, mdn.days, seconds, f'.{nanoseconds:09}')
    return value if mdn.months else value.replace('P', 'P0M')


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
