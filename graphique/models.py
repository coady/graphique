import base64
import types
from datetime import date, datetime, time
from decimal import Decimal
from typing import List, NewType, Optional
import numpy as np
import pyarrow as pa
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


class Query:
    """base class for predicates"""

    locals().update(dict.fromkeys(ops, undefined))

    def asdict(self):
        return {name: value for name, value in self.__dict__.items() if value is not undefined}


@strawberry.input(description="predicates for booleans")
class BooleanQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bool])


@strawberry.input(description="predicates for ints")
class IntQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[int])
    isin: Optional[List[int]] = undefined


@strawberry.input(description="predicates for longs")
class LongQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Long])
    isin: Optional[List[Long]] = undefined


@strawberry.input(description="predicates for floats")
class FloatQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[float])
    isin: Optional[List[float]] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Decimal])
    isin: Optional[List[Decimal]] = undefined


@strawberry.input(description="predicates for dates")
class DateQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[date])
    isin: Optional[List[date]] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[datetime])
    isin: Optional[List[datetime]] = undefined


@strawberry.input(description="predicates for times")
class TimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[time])
    isin: Optional[List[time]] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bytes])
    isin: Optional[List[bytes]] = undefined


@strawberry.input(description="predicates for string")
class StringQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[str])
    isin: Optional[List[str]] = undefined


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

    def length(self) -> Long:
        """number of rows"""
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
Optimized for `null`, and empty queries are implicitly boolean."""
        if query == {'equal': None}:
            return self.array.null_count
        if query == {'not_equal': None}:
            return len(self.array) - self.array.null_count
        return C.count(C.mask(self.array, **query), True)  # type: ignore

    def any(self, **query) -> bool:
        """Return whether any value evaluates to `true`.
Optimized for `null`, and empty queries are implicitly boolean."""
        if query in ({'equal': None}, {'not_equal': None}):
            return bool(resolvers.count(self, **query))
        return C.any(C.mask(self.array, **query))

    def all(self, **query) -> bool:
        """Return whether all values evaluate to `true`.
Optimized for `null`, and empty queries are implicitly boolean."""
        if query in ({'equal': None}, {'not_equal': None}):
            (op,) = {'equal', 'not_equal'} - set(query)
            return not resolvers.count(self, **{op: None})
        return C.all(C.mask(self.array, **query))

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        return C.sum(self.array, exp)

    def min(self):
        """minimum value"""
        return C.min(self.array)

    def max(self):
        """maximum value"""
        return C.max(self.array)

    def quantile(self, q: List[float]) -> List[float]:
        """Return q-th quantiles for values."""
        return np.nanquantile(self.array, q).tolist()

    def sort(self, reverse: bool = False, length: Optional[Long] = None):
        """Return sorted values. Optimized for fixed length."""
        return C.sort(self.array, reverse, length).to_pylist()


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
    length = strawberry.field(resolvers.length, description=resolvers.length.__doc__)
    values = annotate(resolvers.values, List[Optional[bool]])


@strawberry.type(description="column of booleans")
class BooleanColumn:
    Set = BooleanSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BooleanQuery)
    any = query_args(resolvers.any, BooleanQuery)
    all = query_args(resolvers.all, BooleanQuery)
    values = annotate(resolvers.values, List[Optional[bool]])
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="unique ints")
class IntSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length, description=resolvers.length.__doc__)
    values = annotate(resolvers.values, List[Optional[int]])


@strawberry.type(description="column of ints")
class IntColumn:
    Set = IntSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, IntQuery)
    any = query_args(resolvers.any, IntQuery)
    all = query_args(resolvers.all, IntQuery)
    values = annotate(resolvers.values, List[Optional[int]])
    sort = annotate(resolvers.sort, List[Optional[int]])
    sum = annotate(resolvers.sum, int)
    min = annotate(resolvers.min, int)
    max = annotate(resolvers.max, int)
    quantile = strawberry.field(resolvers.quantile, description=resolvers.quantile.__doc__)
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="unique longs")
class LongSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length, description=resolvers.length.__doc__)
    values = annotate(resolvers.values, List[Optional[Long]])


@strawberry.type(description="column of longs")
class LongColumn:
    Set = LongSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, LongQuery)
    any = query_args(resolvers.any, LongQuery)
    all = query_args(resolvers.all, LongQuery)
    values = annotate(resolvers.values, List[Optional[Long]])
    sort = annotate(resolvers.sort, List[Optional[Long]])
    sum = annotate(resolvers.sum, Long)
    min = annotate(resolvers.min, Long)
    max = annotate(resolvers.max, Long)
    quantile = strawberry.field(resolvers.quantile, description=resolvers.quantile.__doc__)
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="column of floats")
class FloatColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, FloatQuery)
    any = query_args(resolvers.any, FloatQuery)
    all = query_args(resolvers.all, FloatQuery)
    values = annotate(resolvers.values, List[Optional[float]])
    sort = annotate(resolvers.sort, List[Optional[float]])
    sum = annotate(resolvers.sum, float)
    min = annotate(resolvers.min, float)
    max = annotate(resolvers.max, float)
    quantile = strawberry.field(resolvers.quantile, description=resolvers.quantile.__doc__)


@strawberry.type(description="column of decimals")
class DecimalColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DecimalQuery)
    any = query_args(resolvers.any, DecimalQuery)
    all = query_args(resolvers.all, DecimalQuery)
    values = annotate(resolvers.values, List[Optional[Decimal]])
    sort = annotate(resolvers.sort, List[Optional[Decimal]])
    min = annotate(resolvers.min, Decimal)
    max = annotate(resolvers.max, Decimal)


@strawberry.type(description="unique dates")
class DateSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length, description=resolvers.length.__doc__)
    values = annotate(resolvers.values, List[Optional[date]])


@strawberry.type(description="column of dates")
class DateColumn:
    Set = DateSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateQuery)
    any = query_args(resolvers.any, DateQuery)
    all = query_args(resolvers.all, DateQuery)
    values = annotate(resolvers.values, List[Optional[date]])
    sort = annotate(resolvers.sort, List[Optional[date]])
    min = annotate(resolvers.min, date)
    max = annotate(resolvers.max, date)
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="column of datetimes")
class DateTimeColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateTimeQuery)
    any = query_args(resolvers.any, DateTimeQuery)
    all = query_args(resolvers.all, DateTimeQuery)
    values = annotate(resolvers.values, List[Optional[datetime]])
    sort = annotate(resolvers.sort, List[Optional[datetime]])
    min = annotate(resolvers.min, datetime)
    max = annotate(resolvers.max, datetime)


@strawberry.type(description="column of times")
class TimeColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, TimeQuery)
    any = query_args(resolvers.any, TimeQuery)
    all = query_args(resolvers.all, TimeQuery)
    values = annotate(resolvers.values, List[Optional[time]])
    sort = annotate(resolvers.sort, List[Optional[time]])
    min = annotate(resolvers.min, time)
    max = annotate(resolvers.max, time)


@strawberry.type(description="column of binaries")
class BinaryColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BinaryQuery)
    any = query_args(resolvers.any, BinaryQuery)
    all = query_args(resolvers.all, BinaryQuery)
    values = annotate(resolvers.values, List[Optional[bytes]])


@strawberry.type(description="unique strings")
class StringSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length, description=resolvers.length.__doc__)
    values = annotate(resolvers.values, List[Optional[str]])


@strawberry.type(description="column of strings")
class StringColumn:
    Set = StringSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, StringQuery)
    any = query_args(resolvers.any, StringQuery)
    all = query_args(resolvers.all, StringQuery)
    values = annotate(resolvers.values, List[Optional[str]])
    sort = annotate(resolvers.sort, List[Optional[str]])
    min = annotate(resolvers.min, str)
    max = annotate(resolvers.max, str)
    unique = annotate(resolvers.unique, Set)


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
