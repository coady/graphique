## Types
A typed schema is automatically generated from the arrow table and its columns. However, advanced usage of tables often creates new columns - or changes the type of existing ones - and therefore falls outside the schema.

### Output
A column within the schema can be accessed by `Table.columns`.
```
{
    columns {
        <name> { ... }
    }
}
```

Any column can be accessed by name using `Table.column` and [inline fragments](https://graphql.org/learn/queries/#inline-fragments).
```
{
    column(name: "...") {
        ... on <Type>Column { ... }
    }
}
```

### Input
Input types don't have the equivalent of inline fragments, but GraphQL is converging on the [tagged union pattern](https://github.com/graphql/graphql-spec/pull/825). Effectively the type of the field becomes the name of the field.

`Dataset.scan` has flexible selection and projection.
```
{
    scan(filter: { ... }, columns: [{ ... }, ...])  { ... }
}
```

`Table.filter` provides a friendlier interface for simple queries on columns within the schema.
```
{
    filter(<name>: { ... }, ...)  { ... }
}
```

`IndexedTable.search` provides the same interface on indexed columns.
```
{
    search(<name>: { ... }, ...)  { ... }
}
```

Note list inputs allow passing a single value, [coercing the input](https://spec.graphql.org/October2021/#sec-List.Input-Coercion) to a list of 1.

## Aggregation
Arrow ListArrays are supported as ListColumns. `Table.group` and `Table.partition` leverage that feature to transform un-grouped columns into ListColumns, which can be accessed via inline fragments and further aggregated. `Table.group` can also aggregate immediately with arrow hash functions. The reason for two different aggregate modes is the trade-off between speed and flexibility. From slowest to fastest:

* `Table.tables` returns a list of tables based on the list scalars.
* `Table.apply(list: {...})` applies general functions to the list scalars.
* `Table.aggregate` applies reduce functions to the list scalars.
* `Table.group(aggregate: {...})` uses arrow hash aggregate functions.

ListColumns support sorting and filtering within their list scalars. They must all have the same value lengths, which is the case when the result of grouping, but list arrays may also be from the original dataset.

## Column selection
Each field resolver transforms a table or array as needed. When working with an embedded library like [pandas](https://pandas.pydata.org), it's common to select a working set of columns for efficiency. Whereas GraphQL has the advantage of knowing the entire query up front, so there is no `Table.select` field because it's done automatically at every level of resolvers.

## Dictionary Arrays
Arrow has dictionary-encoded arrays as a space optimization, but doesn't natively support some builtin functions on them. Support for dictionaries is extended, and often faster by only having to apply functions to the unique values.

## Chunked Arrays
Arrow supports conceptually contiguous arrays in chunks, typically as a result of separate parquet files. Operations are parallelized across chunks when possible. However, grouping and sorting may be memory intensive as they inherently have to combine chunks.

## Nulls
GraphQL continues the long tradition of confusing ["optional" with "nullable"](https://github.com/graphql/graphql-spec/issues/872). Graphique strives to be explicit regarding what may be omitted versus what may be null.

### Output
Arrow has first-class support for nulls, so array scalars are nullable. Non-null scalars are used where relevant.

### Input
Default values and non-null types are used wherever possible. When an input is optional and has no natural default, there's still the issue of distinguishing whether an explicit null input is expected or is semantically different. The input's description field will describe null behavior when expected. Otherwise explicit null behavior is undefined, and assume it errors.
