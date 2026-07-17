"""
GraphQL scalars.
"""

import functools
import keyword
import typing
import warnings
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import ibis.expr.datatypes
import isodate
import pyarrow as pa
import strawberry


def parse_bigint(value) -> int:
    if isinstance(value, int):
        return value
    raise TypeError(f"BigInt cannot represent value: {value}")


def parse_duration(value):
    duration = isodate.parse_duration(value)
    if isinstance(duration, timedelta) and set(value.partition("T")[0]).isdisjoint("YM"):
        return duration
    months = getattr(duration, "years", 0) * 12 + getattr(duration, "months", 0)
    nanoseconds = duration.seconds * 1_000_000_000 + duration.microseconds * 1_000
    return pa.MonthDayNano([months, duration.days, nanoseconds])


duration_isoformat = functools.singledispatch(isodate.duration_isoformat)


@duration_isoformat.register
def _(mdn: pa.MonthDayNano) -> str:
    value = isodate.duration_isoformat(
        isodate.Duration(months=mdn.months, days=mdn.days, microseconds=mdn.nanoseconds // 1_000)
    )
    return value if mdn.months else value.replace("P", "P0M")


BigInt = typing.NewType("BigInt", int)
Duration = typing.NewType("Duration", timedelta)
scalar_map = {
    BigInt: strawberry.scalar(name="BigInt", description="64-bit int", parse_value=parse_bigint),
    Duration: strawberry.scalar(
        name="Duration",
        description="Duration (isoformat)",
        specified_by_url="https://en.wikipedia.org/wiki/ISO_8601#Durations",
        serialize=duration_isoformat,
        parse_value=parse_duration,
    ),
}


def py_type(dt: ibis.DataType) -> type | typing.NewType | None:
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
            return strawberry.scalars.Base64
        case ibis.expr.datatypes.String() | ibis.expr.datatypes.UUID():
            return str
        case ibis.expr.datatypes.Array():
            return list
        case ibis.expr.datatypes.Struct():
            return strawberry.scalars.JSON


def schema_types(schema: ibis.Schema, *, filters: bool = False) -> Iterator:
    """Generate name and py types from schema.

    Args:
        filters: only yield filterable columns; arrays become `list[element]`
    """
    for name in schema:
        if not name.isidentifier() or keyword.iskeyword(name):
            warnings.warn(f"invalid field name: {name}")
            continue
        scalar = py_type(schema[name])
        if scalar is None:
            warnings.warn(f"unknown data type: {schema[name]}")
            continue
        elif scalar is strawberry.scalars.JSON and filters:
            continue
        elif scalar is list and filters:
            scalar = py_type(schema[name].value_type)
            if scalar in (None, list, strawberry.scalars.JSON):
                continue
            scalar = list[scalar]
        yield name, scalar
