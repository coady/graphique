"""
GraphQL input types.
"""
import functools
import inspect
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, List, Optional
import strawberry
from strawberry import UNSET
from strawberry.annotation import StrawberryAnnotation
from strawberry.arguments import StrawberryArgument
from strawberry.field import StrawberryField
from strawberry.types.fields.resolver import StrawberryResolver
from typing_extensions import Annotated
from .core import ListChunk, Column
from .scalars import Long, classproperty

comparisons = ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal')
link = 'https://arrow.apache.org/docs/python/api/compute.html'


class Input:
    """Common utilities for input types."""

    nullables: dict = {}

    def __init_subclass__(cls):
        for name in set(cls.nullables) & set(cls.__annotations__):
            setattr(cls, name, default_field(description=cls.nullables[name]))

    def keys(self):
        for name, value in self.__dict__.items():
            if value is None and name not in self.nullables:
                raise TypeError(f"`{self.__class__.__name__}.{name}` is optional, not nullable")
            if value is not UNSET:
                yield name

    def __getitem__(self, name):
        value = getattr(self, name)
        return dict(value) if hasattr(value, 'keys') else value

    @classproperty
    def resolver(cls) -> Callable:
        """a decorator which flattens an input type's fields into arguments"""
        annotations = dict(cls.__annotations__)
        defaults = {name: getattr(cls, name) for name in annotations if hasattr(cls, name)}
        for field in cls._type_definition.fields:  # type: ignore
            if not hasattr(cls, field.name):
                argument = strawberry.argument(description=field.description)
                annotations[field.name] = Annotated[annotations[field.name], argument]
                defaults[field.name] = field.default_factory()
        for name in cls.nullables:
            argument = strawberry.argument(description=cls.nullables[name])
            annotations[name] = Annotated[annotations[name], argument]
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


def resolve_annotations(func: Callable, annotations: dict, defaults: dict = {}) -> StrawberryField:
    """Return field by transforming annotations into function arguments."""
    resolver = StrawberryResolver(func)
    resolver.arguments = [
        StrawberryArgument(
            python_name=name,
            graphql_name=None,
            type_annotation=StrawberryAnnotation(annotation),
            default=defaults.get(name, UNSET),
        )
        for name, annotation in annotations.items()
    ]
    return StrawberryField(
        python_name=func.__name__,
        type_annotation=StrawberryAnnotation(func.__annotations__['return']),
        description=inspect.getdoc(func),
        base_resolver=resolver,
    )


def default_field(default_factory: Callable = lambda: UNSET, **kwargs) -> StrawberryField:
    """Use dataclass `default_factory` for `UNSET` or mutables."""
    return strawberry.field(default_factory=default_factory, **kwargs)


@strawberry.input(
    description=f"nominal [predicates]({link}#comparisons) projected across two columns"
)
class NominalFilter(Input):
    equal: Optional[str] = UNSET
    not_equal: Optional[str] = UNSET


@strawberry.input(
    description=f"ordinal [predicates]({link}#comparisons) projected across two columns"
)
class OrdinalFilter(NominalFilter):
    less: Optional[str] = UNSET
    less_equal: Optional[str] = UNSET
    greater: Optional[str] = UNSET
    greater_equal: Optional[str] = UNSET


class Query(Input):
    """base class for predicates"""

    type_map: dict
    locals().update(dict.fromkeys(comparisons, UNSET))
    nullables = {
        'equal': "`null` is equivalent to arrow `is_null`.",
        'not_equal': "`null` is equivalent to arrow `is_valid`.",
    }

    @classmethod
    def annotations(cls, types: dict) -> dict:
        """Return mapping of annotations from a mapping of types."""
        return {
            name: Optional[cls.type_map[types[name]]]
            for name in types
            if types[name] in cls.type_map
        }

    @classmethod
    def resolve_types(cls, types: dict) -> Callable:
        """Return a decorator which transforms the type map into arguments."""
        return functools.partial(resolve_annotations, annotations=cls.annotations(types))


@strawberry.input(description="predicates for booleans")
class BooleanQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bool])


@strawberry.input(description="predicates for ints")
class IntQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[int])
    is_in: Optional[List[int]] = UNSET


@strawberry.input(description="predicates for longs")
class LongQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[Long])
    is_in: Optional[List[Long]] = UNSET


@strawberry.input(description="predicates for floats")
class FloatQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[float])
    is_in: Optional[List[float]] = UNSET


@strawberry.input(description="predicates for decimals")
class DecimalQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[Decimal])
    is_in: Optional[List[Decimal]] = UNSET


@strawberry.input(description="predicates for dates")
class DateQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[date])
    is_in: Optional[List[date]] = UNSET


@strawberry.input(description="predicates for datetimes")
class DateTimeQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[datetime])
    is_in: Optional[List[datetime]] = UNSET


@strawberry.input(description="predicates for times")
class TimeQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[time])
    is_in: Optional[List[time]] = UNSET


@strawberry.input(description="predicates for durations")
class DurationQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[timedelta])


@strawberry.input(description="predicates for binaries")
class Base64Query(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bytes])
    is_in: Optional[List[bytes]] = UNSET


@strawberry.input(description="predicates for strings")
class StringQuery(Query):
    __annotations__ = dict.fromkeys(comparisons, Optional[str])
    is_in: Optional[List[str]] = UNSET


Query.type_map = {
    bool: BooleanQuery,
    int: IntQuery,
    Long: LongQuery,
    float: FloatQuery,
    Decimal: DecimalQuery,
    date: DateQuery,
    datetime: DateTimeQuery,
    time: TimeQuery,
    timedelta: DurationQuery,
    bytes: Base64Query,
    str: StringQuery,
}


@strawberry.input
class Filter(Query):
    name: str
    is_in = UNSET


@strawberry.input(description="predicates for booleans")
class BooleanFilter(Filter):
    __annotations__.update(BooleanQuery.__annotations__)
    apply: NominalFilter = default_field(dict)


@strawberry.input(description="predicates for ints")
class IntFilter(Filter):
    __annotations__.update(IntQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for longs")
class LongFilter(Filter):
    __annotations__.update(LongQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for floats")
class FloatFilter(Filter):
    __annotations__.update(FloatQuery.__annotations__)
    is_finite: bool = False
    is_inf: bool = False
    is_nan: bool = False
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for decimals")
class DecimalFilter(Filter):
    __annotations__.update(DecimalQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for dates")
class DateFilter(Filter):
    __annotations__.update(DateQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for datetimes")
class DateTimeFilter(Filter):
    __annotations__.update(DateTimeQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for times")
class TimeFilter(Filter):
    __annotations__.update(TimeQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for durations")
class DurationFilter(Filter):
    __annotations__.update(DurationQuery.__annotations__)
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for binaries")
class Base64Filter(Filter):
    __annotations__.update(Base64Query.__annotations__)
    apply: NominalFilter = default_field(dict)


@strawberry.input(description=f"[predicates]({link}#string-predicates) for strings")
class StringFilter(Filter):
    __annotations__.update(StringQuery.__annotations__)
    match_substring: Optional[str] = UNSET
    match_like: Optional[str] = UNSET
    ignore_case: bool = strawberry.field(default=False, description="case option for substrings")
    regex: bool = strawberry.field(default=False, description="regex option for substrings")
    string_is_ascii: bool = False
    utf8_is_alnum: bool = False
    utf8_is_alpha: bool = False
    utf8_is_decimal: bool = False
    utf8_is_digit: bool = False
    utf8_is_lower: bool = False
    utf8_is_numeric: bool = False
    utf8_is_printable: bool = False
    utf8_is_space: bool = False
    utf8_is_title: bool = False
    utf8_is_upper: bool = False
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for columns of any type as a tagged union")
class Filters(Input):
    boolean: List[BooleanFilter] = default_field(list)
    int: List[IntFilter] = default_field(list)
    long: List[LongFilter] = default_field(list)
    float: List[FloatFilter] = default_field(list)
    decimal: List[DecimalFilter] = default_field(list)
    date: List[DateFilter] = default_field(list)
    datetime: List[DateTimeFilter] = default_field(list)
    time: List[TimeFilter] = default_field(list)
    duration: List[DurationFilter] = default_field(list)
    binary: List[Base64Filter] = default_field(list)
    string: List[StringFilter] = default_field(list)


@strawberry.input
class Function(Input):
    name: str
    alias: str = ''
    coalesce: Optional[List[str]] = UNSET
    cast: str = strawberry.field(
        default='',
        description="cast array to [arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)",
    )
    fill_null_backward: bool = False
    fill_null_forward: bool = False


@strawberry.input
class OrdinalFunction(Function):
    min_element_wise: Optional[str] = UNSET
    max_element_wise: Optional[str] = UNSET


@strawberry.input
class NumericFunction(OrdinalFunction):
    checked: bool = strawberry.field(default=False, description="check math functions for overlow")
    add: Optional[str] = UNSET
    subtract: Optional[str] = UNSET
    multiply: Optional[str] = UNSET
    divide: Optional[str] = UNSET
    power: Optional[str] = UNSET
    atan2: Optional[str] = UNSET
    abs: bool = False
    negate: bool = False
    sign: bool = False
    ln: bool = False
    log1p: bool = False
    log10: bool = False
    log2: bool = False
    floor: bool = False
    ceil: bool = False
    trunc: bool = False
    round: bool = False
    sin: bool = False
    asin: bool = False
    cos: bool = False
    acos: bool = False
    tan: bool = False
    atan: bool = False
    sqrt: bool = strawberry.field(default=False, description="pyarrow >=8 only")


@strawberry.input(description=f"[functions]({link}#selecting-multiplexing) for booleans")
class BooleanFunction(Function):
    if_else: Optional[List[str]] = UNSET


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for ints")
class IntFunction(NumericFunction):
    bit_wise_or: Optional[str] = UNSET
    bit_wise_and: Optional[str] = UNSET
    bit_wise_xor: Optional[str] = UNSET
    shift_left: Optional[str] = UNSET
    shift_right: Optional[str] = UNSET
    bit_wise_not: bool = False
    fill_null: Optional[int] = UNSET
    digitize: Optional[List[int]] = default_field(description=inspect.getdoc(Column.digitize))


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for longs")
class LongFunction(NumericFunction):
    fill_null: Optional[Long] = UNSET
    digitize: Optional[List[Long]] = default_field(description=inspect.getdoc(Column.digitize))


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for floats")
class FloatFunction(NumericFunction):
    fill_null: Optional[float] = UNSET
    digitize: Optional[List[float]] = default_field(description=inspect.getdoc(Column.digitize))


@strawberry.input(description="functions for decimals")
class DecimalFunction(Function):
    fill_null: Optional[Decimal] = UNSET


@strawberry.input(description="[functions]({link}#temporal-component-extraction) for dates")
class DateFunction(OrdinalFunction):
    fill_null: Optional[date] = UNSET
    years_between: Optional[str] = UNSET
    quarters_between: Optional[str] = UNSET
    weeks_between: Optional[str] = UNSET
    days_between: Optional[str] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
    year: bool = False
    quarter: bool = False
    month: bool = False
    week: bool = False
    us_week: bool = False
    day: bool = False
    day_of_week: bool = False
    day_of_year: bool = False
    strftime: bool = False


@strawberry.input(description="[functions]({link}#temporal-component-extraction) for datetimes")
class DateTimeFunction(OrdinalFunction):
    subtract: Optional[str] = UNSET
    fill_null: Optional[datetime] = UNSET
    years_between: Optional[str] = UNSET
    quarters_between: Optional[str] = UNSET
    weeks_between: Optional[str] = UNSET
    days_between: Optional[str] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
    year: bool = False
    quarter: bool = False
    month: bool = False
    week: bool = False
    us_week: bool = False
    day: bool = False
    day_of_week: bool = False
    day_of_year: bool = False
    hour: bool = False
    minute: bool = False
    second: bool = False
    subsecond: bool = False
    millisecond: bool = False
    microsecond: bool = False
    nanosecond: bool = False
    strftime: bool = False


@strawberry.input(description="[functions]({link}#temporal-component-extraction) for times")
class TimeFunction(OrdinalFunction):
    fill_null: Optional[time] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
    hour: bool = False
    minute: bool = False
    second: bool = False
    subsecond: bool = False
    millisecond: bool = False
    microsecond: bool = False
    nanosecond: bool = False


@strawberry.input(description="functions for durations")
class DurationFunction(Function):
    fill_null: Optional[timedelta] = UNSET


@strawberry.input(description="[functions]({link}#string-transforms) for binaries")
class Base64Function(Function):
    binary_join_element_wise: Optional[List[str]] = UNSET
    fill_null: Optional[bytes] = UNSET
    binary_length: bool = False


@strawberry.input(description="[functions]({link}#string-transforms) for strings")
class StringFunction(Function):
    binary_join_element_wise: Optional[List[str]] = UNSET
    find_substring: Optional[str] = UNSET
    count_substring: Optional[str] = UNSET
    ignore_case: bool = strawberry.field(default=False, description="case option for substrings")
    regex: bool = strawberry.field(default=False, description="regex option for substrings")
    fill_null: Optional[str] = UNSET
    utf8_capitalize: bool = False
    utf8_length: bool = False
    utf8_lower: bool = False
    utf8_upper: bool = False
    utf8_swapcase: bool = False
    utf8_reverse: bool = False


@strawberry.input(description=f"[functions]({link}#selecting-multiplexing) for structs")
class StructFunction(Function):
    case_when: Optional[List[str]] = UNSET


@strawberry.input(description=f"[functions]({link}#structural-transforms) for list")
class ListFunction(Function):
    mode: bool = strawberry.field(default=False, description=inspect.getdoc(ListChunk.mode))
    quantile: bool = strawberry.field(default=False, description=inspect.getdoc(ListChunk.quantile))
    unique: bool = strawberry.field(
        default=False, description="may be faster than `distinct` aggregation"
    )
    value_length: bool = strawberry.field(
        default=False, description="faster than `count` aggregation"
    )


@strawberry.input(description=f"names and optional aliases for [aggregation]({link}#aggregations)")
class Aggregate(Input):
    name: str
    alias: str = ''


@strawberry.input(description=f"options for count [aggregation]({link}#grouped-aggregations)")
class CountAggregate(Aggregate):
    mode: str = 'only_valid'


@strawberry.input(description=f"options for scalar [aggregation]({link}#grouped-aggregations)")
class ScalarAggregate(Aggregate):
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input
class Aggregations(Input):
    all: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.all))
    any: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.any))
    count: List[CountAggregate] = default_field(list, description=inspect.getdoc(ListChunk.count))
    count_distinct: List[CountAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.count_distinct)
    )
    distinct: List[CountAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.distinct)
    )
    first: List[Aggregate] = default_field(list, description=inspect.getdoc(ListChunk.first))
    last: List[Aggregate] = default_field(list, description=inspect.getdoc(ListChunk.last))
    max: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.max))
    mean: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.mean))
    min: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.min))
    one: List[Aggregate] = default_field(list, description=inspect.getdoc(ListChunk.one))
    product: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.product)
    )
    stddev: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.stddev)
    )
    sum: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.sum))
    tdigest: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.tdigest)
    )
    variance: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.variance)
    )


@strawberry.input(description="discrete difference predicates; durations may be in float seconds")
class Diff(Input):
    name: str
    less: Optional[float] = UNSET
    less_equal: Optional[float] = UNSET
    greater: Optional[float] = UNSET
    greater_equal: Optional[float] = UNSET
    nullables = dict.fromkeys(
        ['less', 'less_equal', 'greater', 'greater_equal'],
        "`null` compares the arrays element-wise. A float computes the discrete difference first.",
    )


@strawberry.input(
    description=f"[functions]({link}#arithmetic-functions) projected across two columns"
)
class Projections(Input):
    coalesce: Optional[List[str]] = UNSET
    fill_null: Optional[List[str]] = UNSET
    binary_join_element_wise: Optional[List[str]] = UNSET
    if_else: Optional[List[str]] = UNSET
    case_when: Optional[List[str]] = UNSET
    min_element_wise: Optional[str] = UNSET
    max_element_wise: Optional[str] = UNSET
    add: Optional[str] = UNSET
    subtract: Optional[str] = UNSET
    multiply: Optional[str] = UNSET
    divide: Optional[str] = UNSET
    power: Optional[str] = UNSET
    atan2: Optional[str] = UNSET
    bit_wise_or: Optional[str] = UNSET
    bit_wise_and: Optional[str] = UNSET
    bit_wise_xor: Optional[str] = UNSET
    shift_left: Optional[str] = UNSET
    shift_right: Optional[str] = UNSET
    years_between: Optional[str] = UNSET
    quarters_between: Optional[str] = UNSET
    weeks_between: Optional[str] = UNSET
    days_between: Optional[str] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
