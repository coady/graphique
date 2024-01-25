# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

## [1.5](https://pypi.org/project/graphique/1.5/) - 2024-01-24
### Changed
* Pyarrow >=15 required

### Fixed
* Strawberry >=0.212 compatible
* Starlette >=0.36 compatible

## [1.4](https://pypi.org/project/graphique/1.4/) - 2023-11-05
### Changed
* Pyarrow >=14 required
* Python >=3.9 required
* `group` optimized for memory

### Removed
* `fragments` replaced by `group`
* `min` and `max` replaced by `rank`
* `partition` replaced by `runs`
* `list` aggregation must be explicit
* `group` list functions are in `apply`

## [1.3](https://pypi.org/project/graphique/1.3/) - 2023-08-25
### Changed
* Pyarrow >=13 required
* List filtering and sorting moved to functions and optimized
* Dataset filtering, grouping, and sorting on fragments optimized
* `group` can aggregate entire table

### Added
* `flatten` field for list columns
* `rank` field for min and max filtering
* Schema extensions for metrics and deprecations
* `optional` field for partial query results
* `dropNull`, `fillNull`, and `size` fields
* Command-line utilities
* Allow datasets with invalid field names

### Deprecated
* `fragments` field deprecated and functionality moved to `group` field
* Implicit list aggregation on `group` deprecated
* `partition` field deprecated and renamed to `runs`

## [1.2](https://pypi.org/project/graphique/1.2/) - 2023-05-07
### Changed
* Pyarrow >=12 required
* Grouping fragments optimized
* Group by empty columns
* Batch sorting and grouping into lists

## [1.1](https://pypi.org/project/graphique/1.1/) - 2023-01-29
* Pyarrow >=11 required
* Python >=3.8 required
* Scannable functions added
* List aggregations deprecated
* Group by fragments
* Month day nano interval array
* `min` and `max` fields memory optimized

## [1.0](https://pypi.org/project/graphique/1.0/) - 2022-10-28
* Pyarrow >=10 required
* Dataset schema introspection
* Dataset scanning with selection and projection
* Binary search on sorted columns
* List aggregation, filtering, and sorting optimizations
* Compute functions generalized
* Multiple datasets and federation
* Provisional dataset `join` and `take`

## [0.9](https://pypi.org/project/graphique/0.9/) - 2022-08-04
* Pyarrow >=9 required
* Multi-directional sorting
* Removed unnecessary interfaces
* Filtering has stricter typing

## [0.8](https://pypi.org/project/graphique/0.8/) - 2022-05-08
* Pyarrow >=8 required
* Grouping and aggregation integrated
* `AbstractTable` interface renamed to `Dataset`
* `Binary` scalar renamed to `Base64`

## [0.7](https://pypi.org/project/graphique/0.7/) - 2022-02-04
* Pyarrow >=7 required
* `FILTERS` use query syntax and trigger reading the dataset
* `FEDERATED` field configuration
* List columns support sorting and filtering
* Group by and aggregate optimizations
* Dataset scanning

## [0.6](https://pypi.org/project/graphique/0./) - 2021-10-28
* Pyarrow >=6 required
* Group by optimized and replaced `unique` field
* Dictionary related optimizations
* Null consistency with arrow `count` functions

## [0.5](https://pypi.org/project/graphique/0.5/) - 2021-08-06
* Pyarrow >=5 required
* Stricter validation of inputs
* Columns can be cast to another arrow data type
* Grouping uses large list arrays with 64-bit counts
* Datasets are read on-demand or optionally at startup

## [0.4](https://pypi.org/project/graphique/0.4/) - 2021-05-16
* Pyarrow >=4 required
* `sort` updated to use new native routines
* `partition` tables by adjacent values and differences
* `filter` supports unknown column types using tagged union pattern
* `Groups` replaced with `Table.tables` and `Table.aggregate` fields
* Tagged unions used for `filter`, `apply`, and `partition` functions

## [0.3](https://pypi.org/project/graphique/0.3/) - 2021-01-31
* Pyarrow >=3 required
* `any` and `all` fields
* String column `split` field

## [0.2](https://pypi.org/project/graphique/0.2/) - 2020-11-26
* Pyarrow >= 2 required
* `ListColumn` and `StructColumn` types
* `Groups` type with `aggregate` field
* `group` and `unique` optimized
* Statistical fields: `mode`, `stddev`, `variance`
* `is_in`, `min`, and `max` optimized
