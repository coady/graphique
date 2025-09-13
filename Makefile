check:
	uv run pytest -s --cov

lint:
	uv run ruff check .
	uv run ruff format --check .
	mypy -p graphique

html: docs/schema.md
	PYTHONPATH=$(PWD) uv run mkdocs build

docs/schema.md: docs/schema.graphql
	./node_modules/.bin/graphql-markdown \
		--title "Example Schema" \
		--no-toc \
		--prologue "Generated from a test fixture of zipcodes." \
		$? > $@

docs/schema.graphql: graphique/*.py
	PARQUET_PATH=tests/fixtures/zipcodes.parquet uv run strawberry export-schema graphique.service:app.schema > $@
