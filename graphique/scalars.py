"""
GraphQL scalars.
"""
import base64
import enum
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import NewType
import pyarrow as pa
import strawberry

Long = strawberry.scalar(NewType('Long', int), description="64-bit int")

strawberry.scalar(
    bytes,
    name='Binary',
    description="base64 encoded bytes",
    serialize=lambda b: base64.b64encode(b).decode('utf8'),
    parse_value=base64.b64decode,
)

strawberry.scalar(
    timedelta,
    name='Duration',
    description="duration float (in seconds)",
    serialize=timedelta.total_seconds,
    parse_value=lambda s: timedelta(seconds=s),
)

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

comparisons = {
    'equal': '==',
    'not_equal': '!=',
    'less': '<',
    'less_equal': '<=',
    'greater': '>',
    'greater_equal': '>=',
    'is_in': 'in',
}


@strawberry.enum(description="boolean operator to combine predicates")
class Operator(enum.Enum):
    AND = 'and'
    OR = 'or'
    XOR = 'xor'
    AND_KLEENE = 'and_kleene'
    OR_KLEENE = 'or_kleene'


class classproperty(property):
    """A property bound to a class."""

    def __get__(self, instance, owner):
        return self.fget(owner)
