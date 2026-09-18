check:
	uv run pytest -s --cov

lint:
	uvx ruff check
	uvx ruff format --check
	uv run ty check graphique

html: docs/schema.graphql
	uv run --group docs great-docs build

docs/schema.graphql: graphique/*.py
	PARQUET_PATH=tests/fixtures/zipcodes.parquet uv run strawberry export-schema graphique.service:app.schema > $@
