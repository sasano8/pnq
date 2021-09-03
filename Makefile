all: format test

format: format-black format-isort

format-black:
	@echo [black] && poetry run black .

format-isort:
	@echo [isort] && poetry run isort --profile black --filter-files .

test:
	@echo [pytest] && poetry run pytest .

# documentation:
# 	@rm -rf ./docs/auto
# 	@poetry run sphinx-apidoc --module-first -f -o ./docs/auto ./openapi_client_generator
# 	@poetry run sphinx-build -b singlehtml ./docs ./docs/_build

doc-build:
	@poetry run mkdocs build

doc-serve: doc-build
	# @poetry run mkdocs build
	@poetry run mkdocs serve -a localhost:8001