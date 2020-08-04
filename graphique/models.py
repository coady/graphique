import base64
import types
from datetime import date, datetime, time
from decimal import Decimal
from typing import List, NewType, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from strawberry.types.type_resolver import resolve_type
from strawberry.types.types import ArgumentDefinition, FieldDefinition, undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C


Long = NewType('Long', int)
strawberry.scalar(Long, description="64-bit int")
strawberry.scalar(
    bytes,
    name='Binary',
    description="base64 encoded bytes",
    serialize=lambda b: base64.b64encode(b).decode('utf8'),
    parse_value=base64.b64decode,
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
    pa.lib.Type_FLOAT: float,
    pa.lib.Type_DOUBLE: float,
    pa.lib.Type_DECIMAL: Decimal,
    pa.lib.Type_DATE32: date,
    pa.lib.Type_DATE64: date,
    pa.lib.Type_TIMESTAMP: datetime,
    pa.lib.Type_TIME32: time,
    pa.lib.Type_TIME64: time,
    pa.lib.Type_BINARY: bytes,
    pa.lib.Type_STRING: str,
}
ops = 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'


@strawberry.input(description="predicates projected across two columns of a table")
class Projection:
    __annotations__ = dict.fromkeys(ops, Optional[str])
    locals().update(dict.fromkeys(ops, undefined))

    def asdict(self):
        return {name: value for name, value in self.__dict__.items() if value is not undefined}


class Query:
    """base class for predicates"""

    locals().update(dict.fromkeys(ops, undefined))

    def asdict(self):
        return {
            name: (value.asdict() if hasattr(value, 'asdict') else value)
            for name, value in Projection.asdict(self).items()
        }


@strawberry.input(description="predicates for booleans")
class BooleanQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bool])
    project: Optional[Projection] = undefined


@strawberry.input(description="predicates for ints")
class IntQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[int])
    project: Optional[Projection] = undefined
    is_in: Optional[List[int]] = undefined


@strawberry.input(description="predicates for longs")
class LongQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Long])
    project: Optional[Projection] = undefined
    is_in: Optional[List[Long]] = undefined


@strawberry.input(description="predicates for floats")
class FloatQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[float])
    project: Optional[Projection] = undefined
    is_in: Optional[List[float]] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Decimal])
    project: Optional[Projection] = undefined
    is_in: Optional[List[Decimal]] = undefined


@strawberry.input(description="predicates for dates")
class DateQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[date])
    project: Optional[Projection] = undefined
    is_in: Optional[List[date]] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[datetime])
    project: Optional[Projection] = undefined
    is_in: Optional[List[datetime]] = undefined


@strawberry.input(description="predicates for times")
class TimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[time])
    project: Optional[Projection] = undefined
    is_in: Optional[List[time]] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bytes])
    project: Optional[Projection] = undefined
    is_in: Optional[List[bytes]] = undefined


@strawberry.input(description="predicates for strings")
class StringQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[str])
    project: Optional[Projection] = undefined
    is_in: Optional[List[str]] = undefined
    match_substring: Optional[str] = undefined
    utf8_lower: Optional['StringQuery'] = undefined
    utf8_upper: Optional['StringQuery'] = undefined


query_map = {
    bool: BooleanQuery,
    int: IntQuery,
    Long: LongQuery,
    float: FloatQuery,
    Decimal: DecimalQuery,
    date: DateQuery,
    datetime: DateTimeQuery,
    time: TimeQuery,
    bytes: BinaryQuery,
    str: StringQuery,
}


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value: selections(node) for node in nodes}


class resolvers:
    array: pa.ChunkedArray

    def __init__(self, array, **attrs):
        self.array = array
        self.__dict__.update(attrs)

    @strawberry.field(description="number of rows")
    def length(self) -> Long:
        return len(self.array)  # type: ignore

    def unique(self, info):
        """unique values and counts"""
        if 'counts' in selections(*info.field_nodes):
            values, counts = C.value_counts(self.array).flatten()
        else:
            values, counts = C.unique(self.array), pa.array([])
        return self.Set(values, counts=counts.to_pylist())

    def count(self, **query) -> Long:
        """Return number of matching values.
Optimized for `null`, and empty queries will attempt boolean conversion."""
        if query == {'equal': None}:
            return self.array.null_count
        if query == {'not_equal': None}:
            return len(self.array) - self.array.null_count
        return C.count(C.mask(self.array, **query), True)  # type: ignore

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        return C.sum(self.array, exp)

    @strawberry.field(description="mean of the values")
    def mean(self) -> float:
        return pc.call_function('mean', [self.array]).as_py()

    def min(self):
        """minimum value"""
        return C.min(self.array)

    def max(self):
        """maximum value"""
        return C.max(self.array)

    @strawberry.field(description="Return q-th quantiles for values.")
    def quantile(self, q: List[float]) -> List[float]:
        return np.nanquantile(self.array, q).tolist()

    def sort(self, reverse: bool = False, length: Optional[Long] = None):
        """Return sorted values. Optimized for fixed length."""
        return C.sort(self.array, reverse, length).to_pylist()

    @strawberry.field(description="length of bytes or strings")
    def binary_length(self) -> 'IntColumn':
        return IntColumn(pc.binary_length(self.array))


def annotate(func, return_type):
    clone = types.FunctionType(func.__code__, func.__globals__)
    clone.__annotations__.update(func.__annotations__, **{'return': return_type})
    clone.__defaults__ = func.__defaults__
    return strawberry.field(clone, description=func.__doc__)


def query_args(func, query):
    clone = types.FunctionType(func.__code__, func.__globals__)
    arguments = [
        ArgumentDefinition(name=to_camel_case(name), origin_name=name, type=value, origin=clone)
        for name, value in query.__annotations__.items()
    ]
    for argument in arguments:
        resolve_type(argument)
    clone._field_definition = FieldDefinition(
        name=to_camel_case(func.__name__),
        origin_name=func.__name__,
        type=func.__annotations__['return'],
        origin=clone,
        arguments=arguments,
        description=func.__doc__,
        base_resolver=clone,
    )
    return clone


@strawberry.type(description="unique booleans")
class BooleanSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[bool]])


@strawberry.type(description="column of booleans")
class BooleanColumn:
    Set = BooleanSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BooleanQuery)
    values = annotate(resolvers.values, List[Optional[bool]])
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="unique ints")
class IntSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[int]])


@strawberry.type(description="column of ints")
class IntColumn:
    Set = IntSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, IntQuery)
    values = annotate(resolvers.values, List[Optional[int]])
    sort = annotate(resolvers.sort, List[Optional[int]])
    sum = annotate(resolvers.sum, int)
    mean = resolvers.mean
    min = annotate(resolvers.min, int)
    max = annotate(resolvers.max, int)
    quantile = resolvers.quantile
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="unique longs")
class LongSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[Long]])


@strawberry.type(description="column of longs")
class LongColumn:
    Set = LongSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, LongQuery)
    values = annotate(resolvers.values, List[Optional[Long]])
    sort = annotate(resolvers.sort, List[Optional[Long]])
    sum = annotate(resolvers.sum, Long)
    mean = resolvers.mean
    min = annotate(resolvers.min, Long)
    max = annotate(resolvers.max, Long)
    quantile = resolvers.quantile
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="column of floats")
class FloatColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, FloatQuery)
    values = annotate(resolvers.values, List[Optional[float]])
    sort = annotate(resolvers.sort, List[Optional[float]])
    sum = annotate(resolvers.sum, float)
    mean = resolvers.mean
    min = annotate(resolvers.min, float)
    max = annotate(resolvers.max, float)
    quantile = resolvers.quantile


@strawberry.type(description="column of decimals")
class DecimalColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DecimalQuery)
    values = annotate(resolvers.values, List[Optional[Decimal]])
    sort = annotate(resolvers.sort, List[Optional[Decimal]])
    min = annotate(resolvers.min, Decimal)
    max = annotate(resolvers.max, Decimal)


@strawberry.type(description="unique dates")
class DateSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[date]])


@strawberry.type(description="column of dates")
class DateColumn:
    Set = DateSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateQuery)
    values = annotate(resolvers.values, List[Optional[date]])
    sort = annotate(resolvers.sort, List[Optional[date]])
    min = annotate(resolvers.min, date)
    max = annotate(resolvers.max, date)
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="column of datetimes")
class DateTimeColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateTimeQuery)
    values = annotate(resolvers.values, List[Optional[datetime]])
    sort = annotate(resolvers.sort, List[Optional[datetime]])
    min = annotate(resolvers.min, datetime)
    max = annotate(resolvers.max, datetime)


@strawberry.type(description="column of times")
class TimeColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, TimeQuery)
    values = annotate(resolvers.values, List[Optional[time]])
    sort = annotate(resolvers.sort, List[Optional[time]])
    min = annotate(resolvers.min, time)
    max = annotate(resolvers.max, time)


@strawberry.type(description="column of binaries")
class BinaryColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BinaryQuery)
    values = annotate(resolvers.values, List[Optional[bytes]])
    binary_length = resolvers.binary_length


@strawberry.type(description="unique strings")
class StringSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[str]])


@strawberry.type(description="column of strings")
class StringColumn:
    Set = StringSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, StringQuery)
    values = annotate(resolvers.values, List[Optional[str]])
    sort = annotate(resolvers.sort, List[Optional[str]])
    min = annotate(resolvers.min, str)
    max = annotate(resolvers.max, str)
    unique = annotate(resolvers.unique, Set)
    binary_length = resolvers.binary_length

    @strawberry.field(description="strings converted to lowercase")
    def utf8_lower(self) -> 'StringColumn':
        return StringColumn(pc.utf8_lower(self.array))  # type: ignore

    @strawberry.field(description="strings converted to uppercase")
    def utf8_upper(self) -> 'StringColumn':
        return StringColumn(pc.utf8_upper(self.array))  # type: ignore


column_map = {
    bool: BooleanColumn,
    int: IntColumn,
    Long: LongColumn,
    float: FloatColumn,
    Decimal: DecimalColumn,
    date: DateColumn,
    datetime: DateTimeColumn,
    time: TimeColumn,
    bytes: BinaryColumn,
    str: StringColumn,
}
