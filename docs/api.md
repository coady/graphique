## Types
A typed schema is automatically generated from the arrow table and its columns. However, advanced usage of tables often creates new columns - or changes the type of existing ones - and therefore falls outside the schema. Fields which create columns also allow aliasing, otherwise the column is replaced.

### Output
A column within the schema can be accessed by `Table.columns`.
```
{
    columns {
        <name> { ... }
    }
}
```

Any column can be accessed by name using `Dataset.column` and [inline fragments](https://graphql.org/learn/queries/#inline-fragments).
```
{
    column(name: "...") {
        ... on <Type>Column { ... }
    }
}
```

### Input
Input types don't have the equivalent of inline fragments, but GraphQL is converging on the [OneOf input pattern](https://github.com/graphql/graphql-spec/pull/825). Effectively the type of the field becomes the name of the field.

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

Note list inputs allow passing a single value, [coercing the input](https://spec.graphql.org/October2021/#sec-List.Input-Coercion) to a list of 1.

## Batches
Datasets and scanners are processed in batches when possible, instead of loading the table into memory.

* `group`, `scan`, and `filter` - native parallel batch processing
* `order` with `limit` and `dense`

## Partitions
Partitioned datasets use fragment keys when possible.

* `group` on fragment keys with counts
* `order` with limit on fragment keys

## Column selection
Each field resolver transforms a table or array as needed. When working with an embedded library like [pandas](https://pandas.pydata.org), it's common to select a working set of columns for efficiency. Whereas GraphQL has the advantage of knowing the entire query up front, so there is no `select` field because it's done automatically at every level of resolvers.

## List Arrays
Arrow ListArrays are supported as ListColumns. `group: {aggregate: {list: ...}}` leverages that feature to transform columns into ListColumns, which can be accessed via inline fragments and further aggregated. Though `group` hash aggregate functions are more efficient than creating lists.

## Nulls
GraphQL continues the long tradition of confusing ["optional" with "nullable"](https://github.com/graphql/graphql-spec/issues/872). Graphique strives to be explicit regarding what may be omitted versus what may be null.

### Output
Arrow has first-class support for nulls, so array scalars are nullable. Non-null scalars are used where relevant.

Columns and rows are nullable to allow partial query results. `Dataset.optional` enables [client controlled nullability](https://github.com/graphql/graphql-spec/issues/867).

### Input
Default values and non-null types are used wherever possible. When an input is optional and has no natural default, there are two cases to distinguish:

* if null is expected and semantically different, the input's description explains null behavior
* otherwise the input has an `@optional` directive, and explicit null behavior is undefined
