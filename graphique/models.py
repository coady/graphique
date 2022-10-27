"""
GraphQL output types and resolvers.
"""
import functools
import inspect
import itertools
import types
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, Generic, List, Optional, TypeVar, TYPE_CHECKING
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.field import StrawberryField
from strawberry.types import Info
from typing_extensions import Annotated
from .core import Column as C
from .inputs import links
from .scalars import Duration, Long, type_map

if TYPE_CHECKING:  # pragma: no cover
    from .interface import Dataset
T = TypeVar('T')


def _selections(field):
    for selection in field.selections:
        if hasattr(selection, 'name'):
            yield selection.name
        else:
            yield from _selections(selection)


def selections(*fields) -> set:
    """Return set of field name selections from strawberry `SelectedField`."""
    return set(itertools.chain(*map(_selections, fields)))


def doc_field(func: Optional[Callable] = None, **kwargs: str) -> StrawberryField:
    """Return strawberry field with argument and docstring descriptions."""
    if func is None:
        return functools.partial(doc_field, **kwargs)
    for name in kwargs:
        argument = strawberry.argument(description=kwargs[name])
        func.__annotations__[name] = Annotated[func.__annotations__[name], argument]
    return strawberry.field(func, description=inspect.getdoc(func))


def compute_field(func: Callable):
    """Wrap compute function with its description."""
    doc = inspect.getdoc(getattr(pc, func.__name__))
    return strawberry.field(func, description=doc.splitlines()[0])  # type: ignore


@strawberry.interface(description="column interface")
class Column:
    def __init__(self, array):
        self.array = array

    def __init_subclass__(cls):
        cls.__init__ = cls.__init__

    @strawberry.field(description=links.type)
    def type(self) -> str:
        return str(self.array.type)

    @doc_field
    def length(self) -> Long:
        """array length"""
        return len(self.array)

    @classmethod
    def cast(cls, array: pa.ChunkedArray) -> 'Column':
        """Return typed column based on array type."""
        return cls.type_map[type_map[C.scalar_type(array).id]](array)  # type: ignore

    @classmethod
    def fromscalar(cls, scalar: pa.ListScalar) -> Optional['Column']:
        return None if scalar.values is None else cls.cast(pa.chunked_array([scalar.values]))

    @compute_field
    def count(self, mode: str = 'only_valid') -> Long:
        return pc.count(self.array, mode=mode).as_py()


def annotate(func, return_type, **annotations):
    """Return field from an annotated clone of the function."""
    clone = types.FunctionType(func.__code__, func.__globals__)
    annotations['return'] = return_type
    clone.__annotations__.update(func.__annotations__, **annotations)
    clone.__defaults__ = func.__defaults__
    return strawberry.field(clone, description=inspect.getdoc(func))


@strawberry.type
class NominalColumn(Generic[T], Column):
    @compute_field
    def count_distinct(self, mode: str = 'only_valid') -> Long:
        return pc.count_distinct(self.array, mode=mode).as_py()

    @doc_field
    def value(self, index: Long = 0) -> Optional[T]:
        """scalar value at index"""
        return self.array[index].as_py()

    @doc_field
    def values(self) -> List[Optional[T]]:
        """list of values"""
        return self.array.to_pylist()

    @compute_field
    def drop_null(self) -> List[T]:
        return self.array.drop_null().to_pylist()


@strawberry.type(description="unique values and counts")
class Set(Generic[T]):
    length = doc_field(Column.length)
    values = doc_field(NominalColumn.values)
    counts: List[Long] = strawberry.field(description="list of counts")

    def __init__(self, array, counts=pa.array([])):
        self.array, self.counts = array, counts.to_pylist()


@strawberry.type
class OrdinalColumn(NominalColumn[T]):
    def __init__(self, array):
        super().__init__(array)
        self.min_max = functools.lru_cache(maxsize=None)(functools.partial(C.min_max, array))

    @strawberry.field(description=Set._type_definition.description)  # type: ignore
    def unique(self, info: Info) -> Set[T]:
        if 'counts' in selections(*info.selected_fields):
            return Set(*self.array.value_counts().flatten())
        return Set(self.array.unique())

    @doc_field
    def min(self, skip_nulls: bool = True, min_count: int = 0) -> Optional[T]:
        """minimum value"""
        return self.min_max(skip_nulls=skip_nulls, min_count=min_count)['min']

    @doc_field
    def max(self, skip_nulls: bool = True, min_count: int = 0) -> Optional[T]:
        """maximum value"""
        return self.min_max(skip_nulls=skip_nulls, min_count=min_count)['max']

    def index(self, value, start: Long = 0, end: Optional[Long] = None) -> Long:
        """Find the index of the first occurrence of a given value."""
        return C.index(self.array, value, start, end)


@strawberry.type
class IntervalColumn(OrdinalColumn[T]):
    @compute_field
    def mode(self, n: int = 1, skip_nulls: bool = True, min_count: int = 0) -> Set[T]:
        return Set(*pc.mode(self.array, n, skip_nulls=skip_nulls, min_count=min_count).flatten())

    @compute_field
    def sum(self, skip_nulls: bool = True, min_count: int = 0) -> Optional[T]:
        return pc.sum(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def product(self, skip_nulls: bool = True, min_count: int = 0) -> Optional[T]:
        return pc.product(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def mean(self, skip_nulls: bool = True, min_count: int = 0) -> Optional[float]:
        return pc.mean(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def indices_nonzero(self) -> List[Long]:
        return pc.indices_nonzero(self.array).to_pylist()


@strawberry.type
class RatioColumn(IntervalColumn[T]):
    @compute_field
    def stddev(self, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0) -> Optional[float]:
        return pc.stddev(self.array, ddof=ddof, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def variance(
        self, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0
    ) -> Optional[float]:
        options = {'skip_nulls': skip_nulls, 'min_count': min_count}
        return pc.variance(self.array, ddof=ddof, **options).as_py()

    @compute_field
    def quantile(
        self,
        q: List[float] = [0.5],
        interpolation: str = 'linear',
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> List[Optional[float]]:
        options = {'skip_nulls': skip_nulls, 'min_count': min_count}
        return pc.quantile(self.array, q=q, interpolation=interpolation, **options).to_pylist()

    @compute_field
    def tdigest(
        self,
        q: List[float] = [0.5],
        delta: int = 100,
        buffer_size: int = 500,
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> List[Optional[float]]:
        options = {'buffer_size': buffer_size, 'skip_nulls': skip_nulls, 'min_count': min_count}
        return pc.tdigest(self.array, q=q, delta=delta, **options).to_pylist()


def generic_type(cls, **kwargs):
    return lambda tp: StrawberryAnnotation(strawberry.type(tp, **kwargs)[cls]).resolve()


@generic_type(bool, name='eanColumn', description="column of booleans")
class BooleanColumn(IntervalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=bool)

    @compute_field
    def any(self, skip_nulls: bool = True, min_count: int = 1) -> Optional[bool]:
        return pc.any(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def all(self, skip_nulls: bool = True, min_count: int = 1) -> Optional[bool]:
        return pc.all(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()


@generic_type(int, name='Column', description="column of ints")
class IntColumn(RatioColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=int)

    @doc_field
    def take_from(
        self, info: Info, field: str
    ) -> Annotated['Dataset', strawberry.lazy('.interface')]:
        """Provisional: select indices from a table on the root Query type."""
        root = getattr(info.root_value, field)
        return type(root)(root.scanner(info).take(self.array.combine_chunks()))


@generic_type(Long, name='Column', description="column of longs")
class LongColumn(RatioColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=Long)
    take_from = doc_field(IntColumn.take_from)


@generic_type(float, name='Column', description="column of floats")
class FloatColumn(RatioColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=float)


@generic_type(Decimal, name='Column', description="column of decimals")
class DecimalColumn(IntervalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=Decimal)


@generic_type(date, name='Column', description="column of dates")
class DateColumn(OrdinalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=date)


@generic_type(datetime, name='Column', description="column of datetimes")
class DatetimeColumn(OrdinalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=datetime)


@generic_type(time, name='Column', description="column of times")
class TimeColumn(OrdinalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=time)


@generic_type(Duration, name='Column', description="column of durations")
class DurationColumn(NominalColumn[T]):
    ...


@generic_type(strawberry.scalars.Base64, name='Column', description="column of binaries")
class Base64Column(OrdinalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=bytes)


@generic_type(str, name='ingColumn', description="column of strings")
class StringColumn(OrdinalColumn[T]):
    index = annotate(OrdinalColumn.index, Long, value=str)


@strawberry.type(description="column of lists")
class ListColumn(Column):
    @doc_field
    def value(self, index: Long = 0) -> Optional[Column]:
        """scalar column at index"""
        return self.fromscalar(self.array[index])

    @doc_field
    def values(self) -> List[Optional[Column]]:
        """list of columns"""
        return list(map(self.fromscalar, self.array))

    @doc_field
    def flatten(self) -> Column:
        """concatenation of all sub-lists"""
        return self.cast(pc.list_flatten(self.array))


@strawberry.type(description="column of structs")
class StructColumn(Column):
    value = annotate(NominalColumn.value, Optional[dict])

    @doc_field
    def names(self) -> List[str]:
        """field names"""
        return [field.name for field in self.array.type]

    @doc_field(name="field name(s); multiple names access nested fields")
    def column(self, name: List[str]) -> Column:
        """Return struct field as a column."""
        dataset = ds.dataset(pa.table({'': self.array}))
        return self.cast(*dataset.to_table(columns={'': pc.field('', *name)}))


Column.type_map = {  # type: ignore
    bool: BooleanColumn,
    int: IntColumn,
    Long: LongColumn,
    float: FloatColumn,
    Decimal: DecimalColumn,
    date: DateColumn,
    datetime: DatetimeColumn,
    time: TimeColumn,
    timedelta: DurationColumn,
    bytes: Base64Column,
    str: StringColumn,
    list: ListColumn,
    dict: StructColumn,
}
