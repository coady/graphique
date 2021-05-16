"""
GraphQL input types.
"""
import functools
import inspect
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, Iterator, List, Optional
import strawberry
from strawberry.arguments import get_arguments_from_annotations
from strawberry.field import StrawberryField
from strawberry.types.fields.resolver import StrawberryResolver
from strawberry.types.types import undefined
from typing_extensions import Annotated
from .core import Column
from .scalars import Long, classproperty

ops = ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal')
link = 'https://arrow.apache.org/docs/python/api/compute.html'


class Input:
    """Common utilities for input types."""

    def keys(self):
        return [name for name in self.__dict__ if getattr(self, name) is not undefined]

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
    def resolver(cls) -> Callable:
        """a decorator which transforms the subclass input types into arguments"""
        annotations = {}
        for subclass in cls.subclasses():
            name = subclass.__name__.split(cls.__name__)[0].lower()  # type: ignore
            argument = strawberry.argument(description=subclass._type_definition.description)
            annotations[name] = Annotated[List[subclass], argument]  # type: ignore
        defaults = dict.fromkeys(annotations, [])  # type: dict
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


def resolve_annotations(func: Callable, annotations: dict, defaults: dict = {}) -> StrawberryField:
    """Return field by transforming annotations into function arguments."""
    kind = inspect.Parameter.KEYWORD_ONLY
    parameters = {
        name: inspect.Parameter(name, kind, default=defaults.get(name, undefined))
        for name in annotations
    }
    resolver = StrawberryResolver(func)
    resolver.arguments = get_arguments_from_annotations(annotations, parameters, func)
    return StrawberryField(
        python_name=func.__name__,
        graphql_name='',
        type_=func.__annotations__['return'],
        description=func.__doc__,
        base_resolver=resolver,
    )


@strawberry.input(
    description=f"nominal [predicates]({link}#comparisons) projected across two columns"
)
class NominalFilter(Input):
    equal: Optional[str] = undefined
    not_equal: Optional[str] = undefined


@strawberry.input(
    description=f"ordinal [predicates]({link}#comparisons) projected across two columns"
)
class OrdinalFilter(NominalFilter):
    less: Optional[str] = undefined
    less_equal: Optional[str] = undefined
    greater: Optional[str] = undefined
    greater_equal: Optional[str] = undefined


class Query(Input):
    """base class for predicates"""

    type_map: dict
    locals().update(dict.fromkeys(ops, undefined))

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
        annotations = cls.__annotations__
        defaults = {name: getattr(cls, name) for name in annotations}
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


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


Query.type_map = {  # type: ignore
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
    is_in = undefined


@strawberry.input(description="predicates for booleans")
class BooleanFilter(Filter):
    __annotations__.update(BooleanQuery.__annotations__)  # type: ignore
    apply: Optional[NominalFilter] = undefined


@strawberry.input(description="predicates for ints")
class IntFilter(Filter):
    __annotations__.update(IntQuery.__annotations__)  # type: ignore
    absolute: bool = False
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for longs")
class LongFilter(Filter):
    __annotations__.update(LongQuery.__annotations__)  # type: ignore
    absolute: bool = False
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for floats")
class FloatFilter(Filter):
    __annotations__.update(FloatQuery.__annotations__)  # type: ignore
    absolute: bool = False
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalFilter(Filter):
    __annotations__.update(DecimalQuery.__annotations__)  # type: ignore
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for dates")
class DateFilter(Filter):
    __annotations__.update(DateQuery.__annotations__)  # type: ignore
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeFilter(Filter):
    __annotations__.update(DateTimeQuery.__annotations__)  # type: ignore
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for times")
class TimeFilter(Filter):
    __annotations__.update(TimeQuery.__annotations__)  # type: ignore
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for durations")
class DurationFilter(Filter):
    __annotations__.update(DurationQuery.__annotations__)  # type: ignore
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryFilter(Filter):
    __annotations__.update(BinaryQuery.__annotations__)  # type: ignore
    apply: Optional[NominalFilter] = undefined


@strawberry.input(description=f"[predicates]({link}#string-predicates) for strings")
class StringFilter(Filter):
    __annotations__.update(StringQuery.__annotations__)  # type: ignore
    match_substring: Optional[str] = undefined
    match_substring_regex: Optional[str] = undefined
    utf8_lower: bool = False
    utf8_upper: bool = False
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
    apply: Optional[OrdinalFilter] = undefined


@strawberry.input(description="predicates for columns of any type as a tagged union")
class Filters(Input):
    boolean: List[BooleanFilter] = ()  # type: ignore
    int: List[IntFilter] = ()  # type: ignore
    long: List[LongFilter] = ()  # type: ignore
    float: List[FloatFilter] = ()  # type: ignore
    decimal: List[DecimalFilter] = ()  # type: ignore
    date: List[DateFilter] = ()  # type: ignore
    datetime: List[DateTimeFilter] = ()  # type: ignore
    time: List[TimeFilter] = ()  # type: ignore
    duration: List[DurationFilter] = ()  # type: ignore
    binary: List[BinaryFilter] = ()  # type: ignore
    string: List[StringFilter] = ()  # type: ignore


def doc_func(func, default=undefined):
    return strawberry.field(default=default, description=func.__doc__)


@strawberry.input
class Function(Input):
    name: str
    alias: str = ''


@strawberry.input
class OrdinalFunction(Function):
    minimum: Optional[str] = doc_func(Column.minimum)
    maximum: Optional[str] = doc_func(Column.maximum)


@strawberry.input
class NumericFunction(OrdinalFunction):
    add: Optional[str] = undefined
    subtract: Optional[str] = undefined
    multiply: Optional[str] = undefined
    divide: Optional[str] = undefined
    power: Optional[str] = undefined
    absolute: bool = doc_func(Column.absolute, default=False)


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for ints")
class IntFunction(NumericFunction):
    fill_null: Optional[int] = undefined
    digitize: Optional[List[int]] = doc_func(Column.digitize)


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for longs")
class LongFunction(NumericFunction):
    fill_null: Optional[Long] = undefined
    digitize: Optional[List[Long]] = doc_func(Column.digitize)


@strawberry.input(description=f"[functions]({link}#arithmetic-functions) for floats")
class FloatFunction(NumericFunction):
    fill_null: Optional[float] = undefined
    digitize: Optional[List[float]] = doc_func(Column.digitize)


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
    fill_null: Optional[bytes] = undefined
    binary_length: bool = False


@strawberry.input(description="functions for strings")
class StringFunction(OrdinalFunction):
    fill_null: Optional[str] = undefined
    utf8_length: bool = False
    utf8_lower: bool = False
    utf8_upper: bool = False


@strawberry.input(description=f"names and optional aliases for [aggregation]({link}#aggregations)")
class Field(Input):
    name: str
    alias: str = ''


@strawberry.input(description="a scalar compared to discrete differences")
class DiffScalar(Input):
    int: Optional[int]
    long: Optional[Long]
    float: Optional[float]
    datetime: Optional[timedelta] = undefined
    float = long = int = undefined  # defaults here because of an obscure dataclass bug


@strawberry.input(description="discrete difference predicates")
class Diff(Input):
    name: str
    less: Optional[DiffScalar] = undefined
    less_equal: Optional[DiffScalar] = undefined
    greater: Optional[DiffScalar] = undefined
    greater_equal: Optional[DiffScalar] = undefined


@strawberry.input(
    description=f"[functions]({link}#arithmetic-functions) projected across two columns"
)
class Projections(Input):
    minimum: Optional[str] = doc_func(Column.minimum)
    maximum: Optional[str] = doc_func(Column.maximum)
    add: Optional[str] = undefined
    subtract: Optional[str] = undefined
    multiply: Optional[str] = undefined
    divide: Optional[str] = undefined
    power: Optional[str] = undefined
