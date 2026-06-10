
# Makefile: build LaTeX PDF for doc/sporf.tex

.PHONY: all clean

TEX = doc/sporf.tex
OUT = doc/sporf.pdf

all: $(OUT)

$(OUT): $(TEX)
	@echo "Building $(OUT) from $(TEX)"
	@if command -v tectonic >/dev/null 2>&1; then \
		(cd doc && tectonic sporf.tex); \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -silent $(TEX); \
	else \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=doc $(TEX); \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=doc $(TEX); \
	fi

clean:
	@echo "Removing LaTeX intermediate files and PDFs in doc/"
	@rm -f doc/*.aux doc/*.log doc/*.out doc/*.toc doc/*.fls doc/*.fdb_latexmk doc/*.synctex.gz doc/*.pdf
