all: format test

format: format-black format-isort

format-black:
	@echo [black] && poetry run black . --exclude "pnq\/__template__\.py"

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
	@cat docs/index.md > README.md
	@echo "" >> README.md
	@echo "" >> README.md
	@cat CHANGELOG.md >> README.md

doc-serve: doc-build
	# @poetry run mkdocs build
	@poetry run mkdocs serve -a localhost:8001


generate:
	@poetry run python3 pnq_template/generate.py -i pnq/__template__.py -o pnq/types.py
	@make format