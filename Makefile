check:
	python -m pytest -s --cov

lint:
	black --check .
	ruff .
	mypy -p graphique

html: docs/schema.md
	PYTHONPATH=$(PWD) python -m mkdocs build

docs/schema.md: docs/schema.graphql
	./node_modules/.bin/graphql-markdown \
		--title "Example Schema" \
		--no-toc \
		--prologue "Generated from a test fixture of zipcodes." \
		$? > $@

docs/schema.graphql: graphique/*.py
	PARQUET_PATH=tests/fixtures/zipcodes.parquet strawberry export-schema graphique.service:app.schema > $@
