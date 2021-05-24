## Types
A typed schema is automatically generated from the arrow table and its columns. However, advanced usage of tables often creates new columns - or changes the type of existing ones - and therefore falls outside the schema.

### Output
A column within the schema can be accessed by `Table.columns`.
```
{
    columns {
        $name { ... }
    }
}
```

Any column can be accessed by name using `Table.column` and [inline fragments](https://graphql.org/learn/queries/#inline-fragments).
```
{
    column(name: $name) {
        ... on $TypeColumn { ... }
    }
}
```

### Input
Input types don't have the equivalent of inline fragments, but GraphQL is converging on the [tagged union pattern](https://github.com/graphql/graphql-spec/blob/main/rfcs/InputUnion.md#-problem-sketch) and [tagged types](https://github.com/graphql/graphql-spec/blob/main/rfcs/InputUnion.md#-7-tagged-type). Effectively the type of the field becomes the name of the field.

`IndexedTable.search` allows simple queries on indexed columns.
```
{
    search($name: { ... }, ...)  { ... }
}
```

`Table.filter` allows simple queries on columns within the schema, and extended filters on all columns. The term `on` was chosen because of the similarity to inline fragments.
```
{
    filter(
        query($name: {...}, ...),
        on($type: [{name: $name, ...}, ...], ...),
    )  { ... }
}
```

Note list inputs allow passing a single value, interpreted as a list of 1.
```
on($type: {name: $name, ...}, ...)
```

## Aggregation
Arrow ListArrays are supported as ListColumns. `Table.group` and `Table.partition` leverage that feature to transform the ungrouped columns into ListColumns, which can be accessed via inline fragments. `Table.tables` returns a list of tables based on the list scalars; it's flexible but the slowest option. `Table.aggregate` applies reduce functions to the ListColumns, which is faster.

Fastest of all, is not needing all the values of a list scalar. `Table.unique` also groups but only returns scalars: the keys, first value, last value, and count. When only `min` and `max` values are needed, it may be faster to `sort` first, just to use `unique`.

`Table.group` and `Table.unique` have optimized C++ implementations at the array level.

## Column selection
Each field resolver transforms a table or array as needed. When working with an embedded library like [pandas](https://pandas.pydata.org), it's common to select a working set of columns for efficiency. Whereas GraphQL has the advantage of knowing the entire query up front, so there is no `Table.select` field because it's done automatically at every level of resolvers.

## Dictionary Arrays
Arrow has dictionary-encoded arrays as a space optimization, but doesn't natively support many builtin functions on them. Support for dictionaries is extended, and often faster by only having to apply functions to the unique values.

## Chunked Arrays
Arrow supports conceptually contiguous arrays in chunks, typically as a result of separate parquet files. Operations are parallelized across chunks when possible. However, grouping and sorting may be memory intensive as they inherently have to combine chunks.
