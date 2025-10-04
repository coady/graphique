"""
GraphQL scalars.
"""

import functools
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import ibis
import isodate
import pyarrow as pa
import strawberry


def parse_bigint(value) -> int:
    if isinstance(value, int):
        return value
    raise TypeError(f"BigInt cannot represent value: {value}")


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


BigInt = strawberry.scalar(int, name='BigInt', description="64-bit int", parse_value=parse_bigint)
Duration = strawberry.scalar(
    timedelta | pa.MonthDayNano,
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


def py_type(dt: ibis.DataType) -> type:
    """Return python scalar type from data type."""
    match dt:
        case ibis.expr.datatypes.Boolean():
            return bool
        case ibis.expr.datatypes.Int64():
            return BigInt
        case ibis.expr.datatypes.Integer():
            return int
        case ibis.expr.datatypes.Floating():
            return float
        case ibis.expr.datatypes.Decimal():
            return Decimal
        case ibis.expr.datatypes.Date():
            return date
        case ibis.expr.datatypes.Timestamp():
            return datetime
        case ibis.expr.datatypes.Time():
            return time
        case ibis.expr.datatypes.Interval():
            return Duration
        case ibis.expr.datatypes.Binary():
            return bytes
        case ibis.expr.datatypes.String():
            return str
        case ibis.expr.datatypes.Array():
            return list
        case ibis.expr.datatypes.Struct():
            return dict
    raise TypeError("unknown data type")  # pragma: no cover
