"""
GraphQL scalars.
"""

import functools
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Union
import isodate
import pyarrow as pa
import strawberry


def parse_long(value) -> int:
    if isinstance(value, int):
        return value
    raise TypeError(f"Long cannot represent value: {value}")


def parse_duration(value):
    duration = isodate.parse_duration(value)
    if isinstance(duration, timedelta) and set(value.partition('T')[0]).isdisjoint('YM'):
        return duration
    months = getattr(duration, 'years', 0) * 12 + getattr(duration, 'months', 0)
    nanoseconds = duration.seconds * 1_000_000_000 + duration.microseconds * 1_000
    return pa.MonthDayNano([months, duration.days, nanoseconds])


duration_isoformat = functools.singledispatch(isodate.duration_isoformat)


@duration_isoformat.register
def _(mdn: pa.MonthDayNano) -> str:
    value = isodate.duration_isoformat(
        isodate.Duration(months=mdn.months, days=mdn.days, microseconds=mdn.nanoseconds // 1_000)
    )
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


def py_type(dt: pa.DataType) -> type:
    return type_map[(dt.value_type if pa.types.is_dictionary(dt) else dt).id]
