"""
GraphQL input types.
"""
import functools
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, Iterator, List, Optional, no_type_check
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.arguments import StrawberryArgument, UNSET
from strawberry.field import StrawberryField
from strawberry.types.fields.resolver import StrawberryResolver
from typing_extensions import Annotated
from .core import Column
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

    @classmethod
    def subclasses(cls) -> Iterator:
        """Generate subclasses with a description for annotating."""
        for subclass in cls.__subclasses__():
            if subclass._type_definition.description:  # type: ignore
                yield subclass
            yield from subclass.subclasses()

    @classproperty
    @no_type_check
    def resolver(cls) -> Callable:
        """a decorator which transforms the subclass input types into arguments"""
        annotations = {}
        for subclass in cls.subclasses():
            name = subclass.__name__.split(cls.__name__)[0].lower()
            argument = strawberry.argument(description=subclass._type_definition.description)
            annotations[name] = Annotated[List[subclass], argument]
        defaults: dict = dict.fromkeys(annotations, [])
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
        description=func.__doc__,
        base_resolver=resolver,
    )


def default_field(default_factory: Callable = lambda: UNSET, **kwargs) -> StrawberryField:
    """Use dataclass `default_factory` for GraphQL `default_value`."""
    field = strawberry.field(default_factory=default_factory, **kwargs)
    field.default_value = default_factory()
    return field


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

    @classproperty
    def resolver(cls) -> Callable:
        """a decorator which transforms the query's fields into arguments"""
        annotations = dict(cls.__annotations__)
        annotations.pop('apply', None)
        defaults = {name: getattr(cls, name) for name in annotations}
        for name in cls.nullables:
            argument = strawberry.argument(description=cls.nullables[name])
            annotations[name] = Annotated[annotations[name], argument]
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


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
    is_in: Optional[List[timedelta]] = UNSET


@strawberry.input(description="predicates for binaries")
class BinaryQuery(Query):
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
    bytes: BinaryQuery,
    str: StringQuery,
}


@strawberry.input
class Filter(Query):
    name: str
    is_in = UNSET


@strawberry.input(description="predicates for booleans")
class BooleanFilter(Filter):
    __annotations__.update(BooleanQuery.__annotations__)  # type: ignore
    apply: NominalFilter = default_field(dict)


@strawberry.input(description="predicates for ints")
class IntFilter(Filter):
    __annotations__.update(IntQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for longs")
class LongFilter(Filter):
    __annotations__.update(LongQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for floats")
class FloatFilter(Filter):
    __annotations__.update(FloatQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for decimals")
class DecimalFilter(Filter):
    __annotations__.update(DecimalQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for dates")
class DateFilter(Filter):
    __annotations__.update(DateQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for datetimes")
class DateTimeFilter(Filter):
    __annotations__.update(DateTimeQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for times")
class TimeFilter(Filter):
    __annotations__.update(TimeQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for durations")
class DurationFilter(Filter):
    __annotations__.update(DurationQuery.__annotations__)  # type: ignore
    apply: OrdinalFilter = default_field(dict)


@strawberry.input(description="predicates for binaries")
class BinaryFilter(Filter):
    __annotations__.update(BinaryQuery.__annotations__)  # type: ignore
    apply: NominalFilter = default_field(dict)


@strawberry.input(description=f"[predicates]({link}#string-predicates) for strings")
class StringFilter(Filter):
    __annotations__.update(StringQuery.__annotations__)  # type: ignore
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
    binary: List[BinaryFilter] = default_field(list)
    string: List[StringFilter] = default_field(list)


@strawberry.input
class Function(Input):
    name: str
    alias: str = ''
    coalesce: Optional[List[str]] = UNSET
    cast: str = default_field(
        str,
        description="cast array to [arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)",
    )


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
    sin: bool = False
    asin: bool = False
    cos: bool = False
    acos: bool = False
    tan: bool = False
    atan: bool = False


@strawberry.input(description=f"[functions]({link}) for booleans")
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
    digitize: Optional[List[int]] = default_field(description=Column.digitize.__doc__)


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for longs")
class LongFunction(NumericFunction):
    fill_null: Optional[Long] = UNSET
    digitize: Optional[List[Long]] = default_field(description=Column.digitize.__doc__)


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for floats")
class FloatFunction(NumericFunction):
    fill_null: Optional[float] = UNSET
    digitize: Optional[List[float]] = default_field(description=Column.digitize.__doc__)


@strawberry.input(description="functions for decimals")
class DecimalFunction(Function):
    pass


@strawberry.input(description="functions for dates")
class DateFunction(OrdinalFunction):
    fill_null: Optional[date] = UNSET


@strawberry.input(description="functions for datetimes")
class DateTimeFunction(OrdinalFunction):
    subtract: Optional[str] = UNSET
    fill_null: Optional[datetime] = UNSET
    year: bool = False
    month: bool = False
    day: bool = False
    day_of_week: bool = False
    day_of_year: bool = False
    hour: bool = False
    minute: bool = False
    second: bool = False
    millisecond: bool = False
    microsecond: bool = False
    nanosecond: bool = False


@strawberry.input(description="functions for times")
class TimeFunction(OrdinalFunction):
    fill_null: Optional[time] = UNSET


@strawberry.input(description="functions for durations")
class DurationFunction(Function):
    fill_null: Optional[timedelta] = UNSET
    abs: bool = False


@strawberry.input(description="functions for binaries")
class BinaryFunction(Function):
    binary_join_element_wise: Optional[List[str]] = UNSET
    fill_null: Optional[bytes] = UNSET
    binary_length: bool = False


@strawberry.input(description="functions for strings")
class StringFunction(Function):
    binary_join_element_wise: Optional[List[str]] = UNSET
    find_substring: Optional[str] = UNSET
    count_substring: Optional[str] = UNSET
    ignore_case: bool = strawberry.field(default=False, description="case option for substrings")
    regex: bool = strawberry.field(default=False, description="regex option for substrings")
    fill_null: Optional[str] = UNSET
    utf8_length: bool = False
    utf8_lower: bool = False
    utf8_upper: bool = False
    utf8_reverse: bool = False


@strawberry.input(description=f"[functions]({link}) for structs")
class StructFunction(Function):
    case_when: Optional[List[str]] = UNSET


@strawberry.input(description=f"names and optional aliases for [aggregation]({link}#aggregations)")
class Field(Input):
    name: str
    alias: str = ''


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
