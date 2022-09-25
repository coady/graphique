"""
GraphQL input types.
"""
import functools
import inspect
import operator
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, Generic, List, Optional, TypeVar, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry
from strawberry import UNSET
from strawberry.annotation import StrawberryAnnotation
from strawberry.arguments import StrawberryArgument
from strawberry.field import StrawberryField
from strawberry.scalars import Base64, JSON
from strawberry.types.fields.resolver import StrawberryResolver
from typing_extensions import Annotated
from .core import ListChunk, Column
from .scalars import Duration, Long, classproperty

T = TypeVar('T')


class links:
    compute = 'https://arrow.apache.org/docs/python/api/compute.html'
    type = '[arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)'


class Input:
    """Common utilities for input types."""

    nullables: set = set()

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
            argument = strawberry.argument(description=field.description)
            annotations[field.name] = Annotated[annotations[field.name], argument]
            defaults[field.name] = field.default_factory()
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


def resolve_annotations(func: Callable, annotations: dict, defaults: dict = {}) -> StrawberryField:
    """Return field by transforming annotations into function arguments."""
    resolver = StrawberryResolver(func)
    resolver.arguments = [
        StrawberryArgument(
            python_name=name,
            graphql_name=name,
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


def default_field(
    default_factory: Callable = lambda: UNSET, func: Callable = None, **kwargs
) -> StrawberryField:
    """Use dataclass `default_factory` for `UNSET` or mutables."""
    if func is not None:
        kwargs['description'] = inspect.getdoc(func).splitlines()[0]  # type: ignore
    return strawberry.field(default_factory=default_factory, **kwargs)


@strawberry.input(description="predicates for scalars")
class Filter(Generic[T], Input):
    eq: Optional[List[Optional[T]]] = default_field(
        description="== or `isin`; `null` is equivalent to arrow `is_null`."
    )
    ne: Optional[T] = default_field(description="!=; `null` is equivalent to arrow `is_valid`.")
    lt: Optional[T] = default_field(description="<")
    le: Optional[T] = default_field(description="<=")
    gt: Optional[T] = default_field(description=r"\>")
    ge: Optional[T] = default_field(description=r"\>=")

    nullables = {'eq', 'ne'}

    @classmethod
    @no_type_check
    def resolve_types(cls, types: dict) -> Callable:
        """Return a decorator which transforms the type map into arguments."""
        defaults = dict.fromkeys(types, {})
        annotations = {name: cls[types[name]] for name in types if types[name] not in (list, dict)}
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


@strawberry.input(description="positional function arguments without scalars")
class Fields:
    name: List[Optional[str]] = strawberry.field(description="column name(s)")
    alias: str = strawberry.field(default='', description="output column name")

    def serialize(self, table):
        """Return (name, args, kwargs) suitable for computing."""
        exclude = {'name', 'alias'}
        return (
            self.alias or self.name[0],
            map(table.column, self.name),
            {name: value for name, value in self.__dict__.items() if name not in exclude},
        )


@strawberry.input(description="positional function arguments with typed scalar")
class Arguments(Generic[T], Fields):
    value: List[Optional[T]] = default_field(list, description="scalar value(s)")
    cast: str = strawberry.field(default='', description=f"cast scalar to {links.type}")

    def serialize(self, table):
        """Return (name, args, kwargs) suitable for computing."""
        exclude = {'name', 'alias', 'value', 'cast'}
        values = (pa.scalar(value, self.cast) if self.cast else value for value in self.value)
        args = [table[name] if name else next(values) for name in self.name]
        return (
            self.alias or next(filter(None, self.name)),
            args + list(values),
            {name: value for name, value in self.__dict__.items() if name not in exclude},
        )


@strawberry.input
class SetLookup(Arguments[T]):
    skip_nulls: bool = False

    def serialize(self, table):
        """Return (name, args, kwargs) suitable for computing."""
        name, (values, *value_set), kwargs = super().serialize(table)
        return name, (values, pa.array(value_set, self.cast or None)), kwargs


@strawberry.input(description=f"applied [functions]({links.compute})")
class Function(Generic[T], Input):
    coalesce: Optional[Arguments[T]] = default_field(func=pc.coalesce)
    fill_null_backward: Optional[Fields] = default_field(func=pc.fill_null_backward)
    fill_null_forward: Optional[Fields] = default_field(func=pc.fill_null_forward)
    if_else: Optional[Arguments[T]] = default_field(func=pc.if_else)
    replace_with_mask: Optional[Fields] = default_field(func=pc.replace_with_mask)
    index_in: Optional[SetLookup[T]] = default_field(func=pc.index_in)


@strawberry.input
class ElementWiseAggregate(Arguments[T]):
    skip_nulls: bool = True


@strawberry.input
class OrdinalFunction(Function[T]):
    min_element_wise: Optional[ElementWiseAggregate[T]] = default_field(func=pc.min_element_wise)
    max_element_wise: Optional[ElementWiseAggregate[T]] = default_field(func=pc.max_element_wise)


@strawberry.input
class Round(Fields):
    ndigits: int = 0
    round_mode: str = 'half_to_even'


@strawberry.input
class RoundToMultiple(Fields):
    multiple: float = 1.0
    round_mode: str = 'half_to_even'


@strawberry.input(
    description="numpy [digitize](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html)"
)
class Digitize(Arguments[T]):
    bins: List[T]
    right: bool = False


@strawberry.input(description=f"arithmetic [functions]({links.compute}#arithmetic-functions)")
class NumericFunction(OrdinalFunction[T]):
    digitize: Optional[Digitize[T]] = default_field(func=Column.digitize)

    add: Optional[Arguments[T]] = default_field(func=pc.power)
    subtract: Optional[Arguments[T]] = default_field(func=pc.subtract)
    multiply: Optional[Arguments[T]] = default_field(func=pc.multiply)
    divide: Optional[Arguments[T]] = default_field(func=pc.divide)
    cumulative_sum: Optional[ElementWiseAggregate[T]] = default_field(func=pc.cumulative_sum)
    cumulative_sum_checked: Optional[ElementWiseAggregate[T]] = default_field(
        func=pc.cumulative_sum_checked
    )

    power: Optional[Arguments[T]] = default_field(func=pc.power)
    power_checked: Optional[Arguments[T]] = default_field(func=pc.power_checked)
    sqrt: Optional[Fields] = default_field(func=pc.sqrt)
    sqrt_checked: Optional[Fields] = default_field(func=pc.sqrt_checked)
    ln: Optional[Fields] = default_field(func=pc.ln)
    ln_checked: Optional[Fields] = default_field(func=pc.ln_checked)
    log1p: Optional[Fields] = default_field(func=pc.log1p)
    log1p_checked: Optional[Fields] = default_field(func=pc.log1p_checked)
    logb: Optional[Arguments[T]] = default_field(func=pc.logb)
    logb_checked: Optional[Arguments[T]] = default_field(func=pc.logb_checked)

    abs: Optional[Fields] = default_field(func=pc.abs)
    abs_checked: Optional[Fields] = default_field(func=pc.abs_checked)
    negate: Optional[Fields] = default_field(func=pc.negate)
    negate_checked: Optional[Fields] = default_field(func=pc.negate_checked)
    sign: Optional[Fields] = default_field(func=pc.sign)
    round: Optional[Round] = default_field(func=pc.round)
    round_to_multiple: Optional[RoundToMultiple] = default_field(func=pc.round_to_multiple)

    sin: Optional[Fields] = default_field(func=pc.sin)
    sin_checked: Optional[Fields] = default_field(func=pc.sin_checked)
    cos: Optional[Fields] = default_field(func=pc.cos)
    cos_checked: Optional[Fields] = default_field(func=pc.cos_checked)
    tan: Optional[Fields] = default_field(func=pc.tan)
    tan_checked: Optional[Fields] = default_field(func=pc.tan_checked)

    asin: Optional[Fields] = default_field(func=pc.asin)
    asin_checked: Optional[Fields] = default_field(func=pc.asin_checked)
    acos: Optional[Fields] = default_field(func=pc.acos)
    acos_checked: Optional[Fields] = default_field(func=pc.acos_checked)
    atan: Optional[Fields] = default_field(func=pc.atan)
    atan2: Optional[Fields] = default_field(func=pc.atan2)


@operator.itemgetter(int)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#arithmetic-functions) for ints"
)
class IntFunction(NumericFunction[T]):
    choose: Optional[Fields] = default_field(func=pc.choose)

    bit_wise_and: Optional[Arguments[T]] = default_field(func=pc.bit_wise_and)
    bit_wise_or: Optional[Arguments[T]] = default_field(func=pc.bit_wise_or)
    bit_wise_xor: Optional[Arguments[T]] = default_field(func=pc.bit_wise_xor)
    bit_wise_not: Optional[Fields] = default_field(func=pc.bit_wise_not)
    shift_left: Optional[Arguments[T]] = default_field(func=pc.shift_left)
    shift_left_checked: Optional[Arguments[T]] = default_field(func=pc.shift_left_checked)
    shift_right: Optional[Arguments[T]] = default_field(func=pc.shift_right)
    shift_right_checked: Optional[Arguments[T]] = default_field(func=pc.shift_right_checked)


@operator.itemgetter(Long)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#arithmetic-functions) for floats"
)
class LongFunction(NumericFunction[T]):
    choose: Optional[Fields] = default_field(func=pc.choose)


@operator.itemgetter(float)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#arithmetic-functions) for floats"
)
class FloatFunction(NumericFunction[T]):
    is_finite: Optional[Fields] = default_field(func=pc.is_finite)
    is_inf: Optional[Fields] = default_field(func=pc.is_inf)
    is_nan: Optional[Fields] = default_field(func=pc.is_nan)
    floor: Optional[Fields] = default_field(func=pc.floor)
    ceil: Optional[Fields] = default_field(func=pc.ceil)
    trunc: Optional[Fields] = default_field(func=pc.trunc)


DecimalFunction = Function[Decimal]


@strawberry.input
class Week(Fields):
    week_starts_monday: bool = True
    count_from_zero: bool = False
    first_week_is_fully_in_year: bool = False


@strawberry.input
class DayOfWeek(Arguments[T]):
    count_from_zero: bool = True
    week_start: int = 1


@strawberry.input
class Strftime(Fields):
    format: str = '%Y-%m-%dT%H:%M:%S'
    locale: str = 'C'


@strawberry.input
class AssumeTimezone(Fields):
    timezone: str
    ambiguous: str = 'raise'
    nonexistent: str = 'raise'


@strawberry.input
class RoundTemporal(Fields):
    multiple: int = 1
    unit: str = 'day'
    week_starts_monday: bool = True
    ceil_is_strictly_greater: bool = False
    calendar_based_origin: bool = False


@strawberry.input
class TemporalFunction(Function[T]):
    ceil_temporal: Optional[RoundTemporal] = default_field(func=pc.ceil_temporal)
    floor_temporal: Optional[RoundTemporal] = default_field(func=pc.floor_temporal)
    round_temporal: Optional[RoundTemporal] = default_field(func=pc.round_temporal)


@operator.itemgetter(date)
@strawberry.input(
    name='Function',
    description=f"[functions]({links.compute}#temporal-component-extraction) for dates",
)
class DateFunction(TemporalFunction[T]):
    strftime: Optional[Strftime] = default_field(func=pc.strftime)
    subtract: Optional[Arguments[T]] = default_field(func=pc.subtract)

    year: Optional[Fields] = default_field(func=pc.year)
    us_year: Optional[Fields] = default_field(func=pc.us_year)
    year_month_day: Optional[Fields] = default_field(func=pc.year_month_day)
    quarter: Optional[Fields] = default_field(func=pc.quarter)
    month: Optional[Fields] = default_field(func=pc.month)
    week: Optional[Week] = default_field(func=pc.week)
    us_week: Optional[Fields] = default_field(func=pc.us_week)
    day: Optional[Fields] = default_field(func=pc.day)
    day_of_week: Optional[DayOfWeek[T]] = default_field(func=pc.day_of_week)
    day_of_year: Optional[Fields] = default_field(func=pc.day_of_year)
    iso_week: Optional[Fields] = default_field(func=pc.iso_week)
    iso_year: Optional[Fields] = default_field(func=pc.iso_year)
    iso_calendar: Optional[Fields] = default_field(func=pc.iso_calendar)
    is_leap_year: Optional[Fields] = default_field(func=pc.is_leap_year)

    years_between: Optional[Arguments[T]] = default_field(func=pc.years_between)
    quarters_between: Optional[Arguments[T]] = default_field(func=pc.quarters_between)
    weeks_between: Optional[DayOfWeek[T]] = default_field(func=pc.weeks_between)
    days_between: Optional[Arguments[T]] = default_field(func=pc.days_between)


@operator.itemgetter(datetime)
@strawberry.input(
    name='Function',
    description=f"[functions]({links.compute}#temporal-component-extraction) for datetimes",
)
class DateTimeFunction(TemporalFunction[T]):
    strftime: Optional[Strftime] = default_field(func=pc.strftime)
    subtract: Optional[Arguments[T]] = default_field(func=pc.subtract)
    assume_timezone: Optional[AssumeTimezone] = default_field(func=pc.assume_timezone)

    year: Optional[Fields] = default_field(func=pc.year)
    us_year: Optional[Fields] = default_field(func=pc.us_year)
    year_month_day: Optional[Fields] = default_field(func=pc.year_month_day)
    quarter: Optional[Fields] = default_field(func=pc.quarter)
    month: Optional[Fields] = default_field(func=pc.month)
    week: Optional[Week] = default_field(func=pc.week)
    us_week: Optional[Fields] = default_field(func=pc.us_week)
    day: Optional[Fields] = default_field(func=pc.day)
    day_of_week: Optional[DayOfWeek[T]] = default_field(func=pc.day_of_week)
    day_of_year: Optional[Fields] = default_field(func=pc.day_of_year)
    iso_week: Optional[Fields] = default_field(func=pc.iso_week)
    iso_year: Optional[Fields] = default_field(func=pc.iso_year)
    iso_calendar: Optional[Fields] = default_field(func=pc.iso_calendar)
    is_leap_year: Optional[Fields] = default_field(func=pc.is_leap_year)

    hour: Optional[Fields] = default_field(func=pc.hour)
    minute: Optional[Fields] = default_field(func=pc.minute)
    second: Optional[Fields] = default_field(func=pc.second)
    subsecond: Optional[Fields] = default_field(func=pc.subsecond)
    millisecond: Optional[Fields] = default_field(func=pc.millisecond)
    microsecond: Optional[Fields] = default_field(func=pc.microsecond)
    nanosecond: Optional[Fields] = default_field(func=pc.nanosecond)

    years_between: Optional[Arguments[T]] = default_field(func=pc.years_between)
    quarters_between: Optional[Arguments[T]] = default_field(func=pc.quarters_between)
    weeks_between: Optional[DayOfWeek[T]] = default_field(func=pc.weeks_between)
    days_between: Optional[Arguments[T]] = default_field(func=pc.days_between)

    hours_between: Optional[Arguments[T]] = default_field(func=pc.hours_between)
    minutes_between: Optional[Arguments[T]] = default_field(func=pc.minutes_between)
    seconds_between: Optional[Arguments[T]] = default_field(func=pc.seconds_between)
    milliseconds_between: Optional[Arguments[T]] = default_field(func=pc.milliseconds_between)
    microseconds_between: Optional[Arguments[T]] = default_field(func=pc.microseconds_between)
    nanoseconds_between: Optional[Arguments[T]] = default_field(func=pc.nanoseconds_between)


@operator.itemgetter(time)
@strawberry.input(
    name='Function',
    description=f"[functions]({links.compute}#temporal-component-extraction) for times",
)
class TimeFunction(TemporalFunction[T]):
    subtract: Optional[Arguments[T]] = default_field(func=pc.subtract)

    hour: Optional[Fields] = default_field(func=pc.hour)
    minute: Optional[Fields] = default_field(func=pc.minute)
    second: Optional[Fields] = default_field(func=pc.second)
    subsecond: Optional[Fields] = default_field(func=pc.subsecond)
    millisecond: Optional[Fields] = default_field(func=pc.millisecond)
    microsecond: Optional[Fields] = default_field(func=pc.microsecond)
    nanosecond: Optional[Fields] = default_field(func=pc.nanosecond)

    hours_between: Optional[Arguments[T]] = default_field(func=pc.hours_between)
    minutes_between: Optional[Arguments[T]] = default_field(func=pc.minutes_between)
    seconds_between: Optional[Arguments[T]] = default_field(func=pc.seconds_between)
    milliseconds_between: Optional[Arguments[T]] = default_field(func=pc.milliseconds_between)
    microseconds_between: Optional[Arguments[T]] = default_field(func=pc.microseconds_between)
    nanoseconds_between: Optional[Arguments[T]] = default_field(func=pc.nanoseconds_between)


DurationFunction = Function[Duration]


@strawberry.input
class Join(Arguments[T]):
    null_handling: str = 'emit_null'
    null_replacement: str = ''


@strawberry.input
class ReplaceSlice(Generic[T], Fields):
    start: int
    stop: int
    replacement: T


@operator.itemgetter(Base64)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#string-transforms) for binaries"
)
class Base64Function(Function[T]):
    binary_join: Optional[Arguments[T]] = default_field(func=pc.binary_join)
    binary_join_element_wise: Optional[Join[T]] = default_field(func=pc.binary_join_element_wise)
    binary_length: Optional[Fields] = default_field(func=pc.binary_length)
    binary_replace_slice: Optional[ReplaceSlice] = default_field(func=pc.binary_replace_slice)
    binary_repeat: Optional[Arguments[int]] = default_field(func=pc.binary_repeat)
    binary_reverse: Optional[Fields] = default_field(func=pc.binary_reverse)


@strawberry.input
class MatchSubstring(Arguments[T]):
    ignore_case: bool = False


@strawberry.input
class Split(Arguments[T]):
    max_splits: Optional[int] = None
    reverse: bool = False


@strawberry.input
class Pad(Fields):
    width: int
    padding: str = ''


@strawberry.input
class ReplaceSubstring(Fields):
    pattern: str
    replacement: str
    max_replacements: Optional[int] = None


@strawberry.input
class Strptime(Fields):
    format: str
    unit: str
    error_is_null: bool = False


@strawberry.input
class Slice(Fields):
    start: int
    stop: Optional[int] = None
    step: int = 1


@operator.itemgetter(str)
@strawberry.input(
    name='ingFunction', description=f"[functions]({links.compute}#string-transforms) for strings"
)
class StringFunction(OrdinalFunction[T]):
    binary_join: Optional[Arguments[T]] = default_field(func=pc.binary_join)
    binary_join_element_wise: Optional[Join[T]] = default_field(func=pc.binary_join_element_wise)
    binary_length: Optional[Fields] = default_field(func=pc.binary_length)

    ends_with: Optional[MatchSubstring[T]] = default_field(func=pc.ends_with)
    starts_with: Optional[MatchSubstring[T]] = default_field(func=pc.starts_with)
    find_substring: Optional[MatchSubstring[T]] = default_field(func=pc.find_substring)
    find_substring_regex: Optional[MatchSubstring[T]] = default_field(func=pc.find_substring_regex)
    count_substring: Optional[MatchSubstring[T]] = default_field(func=pc.count_substring)
    match_substring: Optional[MatchSubstring[T]] = default_field(func=pc.match_substring)
    match_substring_regex: Optional[MatchSubstring[T]] = default_field(
        func=pc.match_substring_regex
    )
    match_like: Optional[MatchSubstring[T]] = default_field(func=pc.match_like)

    utf8_capitalize: Optional[Fields] = default_field(func=pc.utf8_capitalize)
    utf8_length: Optional[Fields] = default_field(func=pc.utf8_length)
    utf8_lower: Optional[Fields] = default_field(func=pc.utf8_lower)
    utf8_upper: Optional[Fields] = default_field(func=pc.utf8_upper)
    utf8_swapcase: Optional[Fields] = default_field(func=pc.utf8_swapcase)
    utf8_title: Optional[Fields] = default_field(func=pc.utf8_title)
    utf8_reverse: Optional[Fields] = default_field(func=pc.utf8_reverse)

    utf8_replace_slice: Optional[ReplaceSlice] = default_field(func=pc.utf8_replace_slice)
    utf8_split_whitespace: Optional[Split] = default_field(func=pc.utf8_split_whitespace)
    split_pattern: Optional[Split] = default_field(func=pc.split_pattern)
    split_pattern_regex: Optional[Split] = default_field(func=pc.split_pattern_regex)
    utf8_ltrim: Optional[Arguments[T]] = default_field(func=pc.utf8_ltrim)
    utf8_ltrim_whitespace: Optional[Fields] = default_field(func=pc.utf8_ltrim_whitespace)
    utf8_rtrim: Optional[Arguments[T]] = default_field(func=pc.utf8_rtrim)
    utf8_rtrim_whitespace: Optional[Fields] = default_field(func=pc.utf8_rtrim_whitespace)
    utf8_trim: Optional[Arguments[T]] = default_field(func=pc.utf8_trim)
    utf8_trim_whitespace: Optional[Fields] = default_field(func=pc.utf8_trim_whitespace)
    utf8_center: Optional[Pad] = default_field(func=pc.utf8_center)
    utf8_lpad: Optional[Pad] = default_field(func=pc.utf8_lpad)
    utf8_rpad: Optional[Pad] = default_field(func=pc.utf8_rpad)
    replace_substring: Optional[ReplaceSubstring] = default_field(func=pc.replace_substring)
    replace_substring_regex: Optional[ReplaceSubstring] = default_field(
        func=pc.replace_substring_regex
    )
    extract_regex: Optional[Arguments[T]] = default_field(func=pc.extract_regex)
    strptime: Optional[Strptime] = default_field(func=pc.strptime)
    utf8_slice_codeunits: Optional[Slice] = default_field(func=pc.utf8_slice_codeunits)

    string_is_ascii: Optional[Fields] = default_field(func=pc.string_is_ascii)
    utf8_is_alnum: Optional[Fields] = default_field(func=pc.utf8_is_alnum)
    utf8_is_alpha: Optional[Fields] = default_field(func=pc.utf8_is_alpha)
    utf8_is_decimal: Optional[Fields] = default_field(func=pc.utf8_is_decimal)
    utf8_is_digit: Optional[Fields] = default_field(func=pc.utf8_is_digit)
    utf8_is_lower: Optional[Fields] = default_field(func=pc.utf8_is_lower)
    utf8_is_numeric: Optional[Fields] = default_field(func=pc.utf8_is_numeric)
    utf8_is_printable: Optional[Fields] = default_field(func=pc.utf8_is_printable)
    utf8_is_space: Optional[Fields] = default_field(func=pc.utf8_is_space)
    utf8_is_title: Optional[Fields] = default_field(func=pc.utf8_is_title)
    utf8_is_upper: Optional[Fields] = default_field(func=pc.utf8_is_upper)


@strawberry.input
class StructField(Fields):
    indices: List[int]


@strawberry.input(description=f"[functions]({links.compute}#selecting-multiplexing) for structs")
class StructFunction(Input):
    fill_null_backward: Optional[Fields] = default_field(func=pc.fill_null_backward)
    fill_null_forward: Optional[Fields] = default_field(func=pc.fill_null_forward)
    case_when: Optional[Fields] = default_field(func=pc.case_when)
    struct_field: Optional[StructField] = default_field(func=pc.struct_field)


@strawberry.input
class Element(Fields):
    index: int


@strawberry.input
class Index(Fields):
    value: JSON
    start: Long = 0
    end: Optional[Long] = None


@strawberry.input
class Mode(Fields):
    n: int = 1
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input
class Quantile(Fields):
    q: List[float] = (0.5,)  # type: ignore
    interpolation: str = 'linear'
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input(description=f"[functions]({links.compute}#structural-transforms) for list")
class ListFunction(Input):
    element: Optional[Element] = default_field(func=ListChunk.element)
    fill_null_backward: Optional[Fields] = default_field(func=pc.fill_null_backward)
    fill_null_forward: Optional[Fields] = default_field(func=pc.fill_null_forward)
    filter: 'Expression' = default_field(dict, description="filter within list scalars")
    index: Optional[Index] = default_field(func=pc.index)
    mode: Optional[Mode] = default_field(func=pc.mode)
    quantile: Optional[Quantile] = default_field(func=pc.quantile)
    value_length: Optional[Fields] = default_field(func=pc.list_value_length)


@strawberry.input(
    description=f"names and optional aliases for [aggregation]({links.compute}#aggregations)"
)
class Aggregate(Input):
    name: str
    alias: str = ''


@strawberry.input(description=f"options for count [aggregation]({links.compute}#aggregations)")
class CountAggregate(Aggregate):
    mode: str = 'only_valid'


@strawberry.input(description=f"options for scalar [aggregation]({links.compute}#aggregations)")
class ScalarAggregate(Aggregate):
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input(description=f"options for variance [aggregation]({links.compute}#aggregations)")
class VarianceAggregate(ScalarAggregate):
    ddof: int = 0


@strawberry.input(description=f"options for tdigest [aggregation]({links.compute}#aggregations)")
class TDigestAggregate(ScalarAggregate):
    q: List[float] = (0.5,)  # type: ignore
    delta: int = 100
    buffer_size: int = 500


@strawberry.input
class Aggregations(Input):
    all: List[ScalarAggregate] = default_field(list, func=pc.all)
    any: List[ScalarAggregate] = default_field(list, func=pc.any)
    approximate_median: List[ScalarAggregate] = default_field(list, func=pc.approximate_median)
    count: List[CountAggregate] = default_field(list, func=pc.count)
    count_distinct: List[CountAggregate] = default_field(list, func=pc.count_distinct)
    distinct: List[CountAggregate] = default_field(
        list, description="distinct values within each scalar"
    )
    first: List[Aggregate] = default_field(list, func=ListChunk.first)
    last: List[Aggregate] = default_field(list, func=ListChunk.last)
    max: List[ScalarAggregate] = default_field(list, func=pc.max)
    mean: List[ScalarAggregate] = default_field(list, func=pc.mean)
    min: List[ScalarAggregate] = default_field(list, func=pc.min)
    min_max: List[ScalarAggregate] = default_field(list, func=pc.min_max)
    one: List[Aggregate] = default_field(list, description="arbitrary value within each scalar")
    product: List[ScalarAggregate] = default_field(list, func=pc.product)
    stddev: List[VarianceAggregate] = default_field(list, func=pc.stddev)
    sum: List[ScalarAggregate] = default_field(list, func=pc.sum)
    tdigest: List[TDigestAggregate] = default_field(list, func=pc.tdigest)
    variance: List[VarianceAggregate] = default_field(list, func=pc.variance)


@strawberry.input(
    description="""Discrete difference predicates.

By default compares by not equal, Specifiying `null` with a predicate compares element-wise.
A float computes the discrete difference first; durations may be in float seconds.
"""
)
class Diff(Input):
    name: str
    less: Optional[float] = default_field(name='lt', description="<")
    less_equal: Optional[float] = default_field(name='le', description="<=")
    greater: Optional[float] = default_field(name='gt', description=r"\>")
    greater_equal: Optional[float] = default_field(name='ge', description=r"\>=")

    nullables = {'less', 'less_equal', 'greater', 'greater_equal'}


@strawberry.input(
    description="""[Dataset expression](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html)
used for [scanning](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html).

Expects one of: a field `name`, a scalar, or an operator with expressions. Single values can be passed for an
[input `List`](https://spec.graphql.org/October2021/#sec-List.Input-Coercion).
* `eq` with a list scalar is equivalent to `isin`
* `eq` with a `null` scalar is equivalent `is_null`
* `ne` with a `null` scalar is equivalent to `is_valid`
"""
)
class Expression:
    name: List[str] = default_field(list, description="field name(s)")
    cast: str = strawberry.field(default='', description=f"cast as {links.type}")
    safe: bool = strawberry.field(default=True, description="check for conversion errors on cast")
    value: Optional[JSON] = default_field(description="JSON scalar; also see typed scalars")

    base64: List[bytes] = default_field(list)
    date_: List[date] = default_field(list, name='date')
    datetime_: List[datetime] = default_field(list, name='datetime')
    decimal: List[Decimal] = default_field(list)
    duration: List[timedelta] = default_field(list)
    time_: List[time] = default_field(list, name='time')

    eq: List['Expression'] = default_field(list, description="==")
    ne: List['Expression'] = default_field(list, description="!=")
    lt: List['Expression'] = default_field(list, description="<")
    le: List['Expression'] = default_field(list, description="<=")
    gt: List['Expression'] = default_field(list, description=r"\>")
    ge: List['Expression'] = default_field(list, description=r"\>=")

    add: List['Expression'] = default_field(list, description=r"\+")
    mul: List['Expression'] = default_field(list, description=r"\*")
    sub: List['Expression'] = default_field(list, description=r"\-")
    truediv: List['Expression'] = default_field(list, name='div', description='/')

    and_: List['Expression'] = default_field(list, name='and', description="&")
    or_: List['Expression'] = default_field(list, name='or', description="|")
    inv: Optional['Expression'] = default_field(description="~")
    and_not: List['Expression'] = default_field(list, func=pc.and_not)
    xor: List['Expression'] = default_field(list, func=pc.xor)
    kleene: bool = strawberry.field(default=False, description="use kleene logic for booleans")

    associatives = ('add', 'mul', 'and_', 'or_', 'xor')
    variadics = ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'sub', 'truediv', 'and_not')
    scalars = ('base64', 'date_', 'datetime_', 'decimal', 'duration', 'time_')

    def to_arrow(self) -> Optional[ds.Expression]:
        """Transform GraphQL expression into a dataset expression."""
        fields = []
        if self.name:
            fields.append(ds.field(*self.name))
        for name in self.scalars:
            scalars = getattr(self, name)
            if self.cast:
                scalars = [pa.scalar(scalar, self.cast) for scalar in scalars]
            if scalars:
                fields.append(scalars[0] if len(scalars) == 1 else scalars)
        if self.value is not UNSET:
            fields.append(self.value)
        for op in self.associatives:
            exprs = [expr.to_arrow() for expr in getattr(self, op)]
            if exprs:
                fields.append(functools.reduce(self.getfunc(op), exprs))
        for op in self.variadics:
            exprs = [expr.to_arrow() for expr in getattr(self, op)]
            if exprs:
                if op == 'eq' and isinstance(exprs[-1], list):
                    field = ds.Expression.isin(*exprs)
                elif exprs[-1] is None and op in ('eq', 'ne'):
                    field, _ = exprs
                    field = field.is_null() if op == 'eq' else field.is_valid()
                else:
                    field = self.getfunc(op)(*exprs)
                fields.append(field)
        if self.inv is not UNSET:
            fields.append(~self.inv.to_arrow())  # type: ignore
        if not fields:
            return None
        if len(fields) > 1:
            raise ValueError(f"conflicting inputs: {', '.join(map(str, fields))}")
        (field,) = fields
        cast = self.cast and isinstance(field, ds.Expression)
        return field.cast(self.cast, self.safe) if cast else field

    def getfunc(self, name):
        if self.kleene:
            name = name.rstrip('_') + '_kleene'
        if name.endswith('_'):  # `and_` and `or_` functions differ from operators
            return getattr(operator, name)
        return getattr(pc if hasattr(pc, name) else operator, name)

    @classmethod
    @no_type_check
    def from_query(cls, **queries: Filter) -> 'Expression':
        """Transform query syntax into an Expression input."""
        exprs = []
        for name, query in queries.items():
            field = cls(name=[name])
            exprs += (cls(**{op: [field, cls(value=value)]}) for op, value in dict(query).items())
        return cls(and_=exprs)


@strawberry.input(description="an `Expression` with an optional alias")
class Projection(Expression):
    alias: str = strawberry.field(default='', description="name of projected column")
