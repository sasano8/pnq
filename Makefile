all: format test

format: format-black format-isort

format-black:
	@echo [black] && poetry run black .

format-isort:
	@echo [isort] && poetry run isort --profile black --filter-files .

test: generate
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


generate:
	@poetry run python3 pnq_template/generate.py -i pnq_template/template.py -o pnq/types.py
	# @poetry run python3 pnq_template/generate.py -i pnq_template/template.jinja2 -o pnq/types.py
	@make format