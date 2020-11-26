"""
GraphQL input types.
"""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import List, Optional
import strawberry
from strawberry.types.types import undefined
from .scalars import Long

ops = 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'


@strawberry.input(description="nominal functions projected across two columns")
class Nominal:
    equal: Optional[str] = undefined
    not_equal: Optional[str] = undefined

    def asdict(self):
        return {name: value for name, value in self.__dict__.items() if value is not undefined}


@strawberry.input(description="ordinal functions projected across two columns")
class Ordinal(Nominal):
    less: Optional[str] = undefined
    less_equal: Optional[str] = undefined
    greater: Optional[str] = undefined
    greater_equal: Optional[str] = undefined
    minimum: Optional[str] = undefined
    maximum: Optional[str] = undefined


@strawberry.input(description="interval functions projected across two columns")
class Interval(Ordinal):
    subtract: Optional[str] = undefined


@strawberry.input(description="ratio functions projected across two columns")
class Ratio(Interval):
    add: Optional[str] = undefined
    multiply: Optional[str] = undefined
    divide: Optional[str] = undefined


class Query:
    """base class for predicates"""

    locals().update(dict.fromkeys(ops, undefined))

    def asdict(self):
        return {
            name: (value.asdict() if hasattr(value, 'asdict') else value)
            for name, value in Nominal.asdict(self).items()
        }


@strawberry.input(description="predicates for booleans")
class BooleanQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bool])


@strawberry.input(description="predicates for ints")
class IntQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[int])
    is_in: Optional[List[int]] = undefined


@strawberry.input(description="predicates for longs")
class LongQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Long])
    is_in: Optional[List[Long]] = undefined


@strawberry.input(description="predicates for floats")
class FloatQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[float])
    is_in: Optional[List[float]] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Decimal])
    is_in: Optional[List[Decimal]] = undefined


@strawberry.input(description="predicates for dates")
class DateQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[date])
    is_in: Optional[List[date]] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[datetime])
    is_in: Optional[List[datetime]] = undefined


@strawberry.input(description="predicates for times")
class TimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[time])
    is_in: Optional[List[time]] = undefined


@strawberry.input(description="predicates for durations")
class DurationQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[timedelta])
    is_in: Optional[List[timedelta]] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bytes])
    is_in: Optional[List[bytes]] = undefined


@strawberry.input(description="predicates for strings")
class StringQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[str])
    is_in: Optional[List[str]] = undefined


query_map = {
    bool: BooleanQuery,
    int: IntQuery,
    Long: LongQuery,
    float: FloatQuery,
    Decimal: DecimalQuery,
    date: DateQuery,
    datetime: DateTimeQuery,
    time: TimeQuery,
    timedelta: DurationQuery,
    bytes: BinaryQuery,
    str: StringQuery,
}


@strawberry.input(description="predicates for booleans")
class BooleanFilter(BooleanQuery):
    apply: Optional[Nominal] = undefined


@strawberry.input(description="predicates for ints")
class IntFilter(IntQuery):
    absolute: Optional[IntQuery] = undefined
    apply: Optional[Ratio] = undefined


@strawberry.input(description="predicates for longs")
class LongFilter(LongQuery):
    absolute: Optional[LongQuery] = undefined
    apply: Optional[Ratio] = undefined


@strawberry.input(description="predicates for floats")
class FloatFilter(FloatQuery):
    absolute: Optional[FloatQuery] = undefined
    apply: Optional[Ratio] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalFilter(DecimalQuery):
    apply: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for dates")
class DateFilter(DateQuery):
    apply: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeFilter(DateTimeQuery):
    duration: Optional[DurationQuery] = undefined
    apply: Optional[Interval] = undefined

    def asdict(self):
        query = super().asdict()
        return dict(query, **query.pop('duration', {}))


@strawberry.input(description="predicates for times")
class TimeFilter(TimeQuery):
    apply: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryFilter(BinaryQuery):
    binary_length: Optional[IntQuery] = undefined
    apply: Optional[Nominal] = undefined


@strawberry.input(description="predicates for strings")
class StringFilter(StringQuery):
    __annotations__ = dict(StringQuery.__annotations__)  # used for `count` interface
    match_substring: Optional[str] = undefined
    binary_length: Optional[IntQuery] = undefined
    utf8_lower: Optional['StringFilter'] = undefined
    utf8_upper: Optional['StringFilter'] = undefined
    string_is_ascii: bool = False
    utf8_is_alnum: bool = False
    utf8_is_alpha: bool = False
    utf8_is_digit: bool = False
    utf8_is_lower: bool = False
    utf8_is_title: bool = False
    utf8_is_upper: bool = False
    apply: Optional[Ordinal] = undefined


filter_map = {
    bool: BooleanFilter,
    int: IntFilter,
    Long: LongFilter,
    float: FloatFilter,
    Decimal: DecimalFilter,
    date: DateFilter,
    datetime: DateTimeFilter,
    time: TimeFilter,
    bytes: BinaryFilter,
    str: StringFilter,
}


@strawberry.input
class Function:
    alias: Optional[str] = undefined
    asdict = Nominal.asdict


@strawberry.input
class OrdinalFunction(Function):
    minimum: Optional[str] = undefined
    maximum: Optional[str] = undefined


@strawberry.input
class RatioFunction(OrdinalFunction):
    add: Optional[str] = undefined
    subtract: Optional[str] = undefined
    multiply: Optional[str] = undefined
    divide: Optional[str] = undefined
    absolute: bool = False


@strawberry.input(description="functions for ints")
class IntFunction(RatioFunction):
    fill_null: Optional[int] = undefined


@strawberry.input(description="functions for longs")
class LongFunction(RatioFunction):
    fill_null: Optional[Long] = undefined


@strawberry.input(description="functions for floats")
class FloatFunction(RatioFunction):
    fill_null: Optional[float] = undefined


@strawberry.input(description="functions for decimals")
class DecimalFunction(OrdinalFunction):
    pass


@strawberry.input(description="functions for dates")
class DateFunction(OrdinalFunction):
    fill_null: Optional[date] = undefined


@strawberry.input(description="functions for datetimes")
class DateTimeFunction(OrdinalFunction):
    subtract: Optional[str] = undefined
    fill_null: Optional[datetime] = undefined


@strawberry.input(description="functions for times")
class TimeFunction(OrdinalFunction):
    fill_null: Optional[time] = undefined


@strawberry.input(description="functions for binaries")
class BinaryFunction(Function):
    binary_length: bool = False


@strawberry.input(description="functions for strings")
class StringFunction(OrdinalFunction):
    binary_length: bool = False
    utf8_lower: bool = False
    utf8_upper: bool = False


@strawberry.input(description="aggregate functions for lists")
class ListFunction(Function):
    count: bool = False
    first: bool = False
    last: bool = False
    unique: bool = False
    min: bool = False
    max: bool = False
    sum: bool = False
    mean: bool = False


function_map = {
    int: IntFunction,
    Long: LongFunction,
    float: FloatFunction,
    Decimal: DecimalFunction,
    date: DateFunction,
    datetime: DateTimeFunction,
    time: TimeFunction,
    bytes: BinaryFunction,
    str: StringFunction,
    list: ListFunction,
}


@strawberry.input(description="names and aliases for aggregation")
class Field:
    name: str
    alias: str = ''


@strawberry.input(description="names and aliases for aggregation of unique values")
class UniqueField(Field):
    count: bool = False
