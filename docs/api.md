## Types
A typed schema is automatically generated from the table and its columns. However, advanced usage of tables often creates new columns - or changes the type of existing ones - and therefore falls outside the schema. Fields which create columns also allow aliasing, otherwise the column is replaced.

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


`Table.filter` provides simple queries for columns within the schema, and a `where` arguments for complex expressions.
```
{
    filter(<name>: { ... }, ..., where: { ... })  { ... }
}
```

`Table.project` also supports complex expressions with aliased column names.
```
{
    project(columns: [{ ... }])  { ... }
}
```

Note list inputs allow passing a single value, [coercing the input](https://spec.graphql.org/October2021/#sec-List.Input-Coercion) to a list of 1.

## Partitions
Partitioned parquet datasets have custom optimization for fragment keys.

* `group` on fragment keys with counts
* `order` with limit on fragment keys

## Column selection
Each field resolver transforms a table or column as needed. Ibis is a [lazily executed](https://ibis-project.org/tutorials/basics), so there is no `select` field because it's handled automatically. Conversely if multiple table fields are requested, the [table is cached](https://ibis-project.org/reference/expression-tables#ibis.expr.types.relations.Table.cache) for performance and consistency.

## Arrays
Ibis [Array columns](https://ibis-project.org/reference/expression-collections#ibis.expr.types.arrays.ArrayValue) are supported. `unnest` flattens arrays back to scalars, and `group: {aggregate: {collect: ...}}` also creates arrays.

## Nulls
GraphQL continues the long tradition of confusing ["optional" with "nullable"](https://github.com/graphql/graphql-spec/issues/872). Graphique strives to be explicit regarding what may be omitted versus what may be null.

### Output
Ibis has first-class support for nulls, so array scalars are nullable. Non-null scalars are used where relevant.

Columns and rows are nullable to allow partial query results. `Dataset.optional` enables [client controlled nullability](https://github.com/graphql/graphql-spec/issues/867).

### Input
Default values and non-null types are used wherever possible. When an input is optional and has no natural default, there are two cases to distinguish:

* if null is expected and semantically different, the input's description explains null behavior
* otherwise the input has an `@optional` directive, and explicit null behavior is undefined
