PYTHON ?= python3

.PHONY: figures publish-figures check-paths pdf

figures: publish-figures

publish-figures:
	$(PYTHON) scripts/publish_figures.py

check-paths:
	$(PYTHON) scripts/check_paths.py

pdf:
	mkdir -p build
	if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
	elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
		( cd build && bibtex main ); \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex; \
	else \
		echo "Missing latexmk or pdflatex/bibtex"; \
		exit 1; \
	fi
