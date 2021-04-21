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
Input types don't have the equivalent of inline fragments, but GraphQL is converging on the [tagged union pattern](https://github.com/graphql/graphql-spec/blob/main/rfcs/InputUnion.md#-problem-sketch) and [tagged types](https://github.com/graphql/graphql-spec/blob/main/rfcs/InputUnion.md#-7-tagged-type). Effectively the type of the field because the name of the field.

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
