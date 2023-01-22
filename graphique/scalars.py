"""
GraphQL scalars.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import pyarrow as pa
import strawberry


def parse_long(value) -> int:
    if isinstance(value, int):
        return value
    raise TypeError(f"Long cannot represent value: {value}")


@strawberry.type(description="month day nano interval")
class Interval:
    months: int = 0
    days: int = 0
    nanoseconds: 'Long' = 0  # type: ignore


Long = strawberry.scalar(int, name='Long', description="64-bit int", parse_value=parse_long)
Duration = strawberry.scalar(  # pragma: no branch
    timedelta,
    name='Duration',
    description="duration float (in seconds)",
    serialize=timedelta.total_seconds,
    parse_value=lambda s: timedelta(seconds=s),
)
scalar_map = {bytes: strawberry.scalars.Base64, dict: strawberry.scalars.JSON, timedelta: Duration}

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
    pa.lib.Type_INTERVAL_MONTH_DAY_NANO: Interval,
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
