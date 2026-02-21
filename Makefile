# GALILEO helper Makefile (paper-facing utilities)

.PHONY: figures-pdf figures-check anonymized-bundle citations-check paper paper-camera-ready paper-clean

# Preflight for SVG->PDF conversion tooling
figures-check:
	./scripts/check_figure_tooling.sh

# Convert docs/paper/figures/*.svg -> paper_figures/pdf/*.pdf
figures-pdf: figures-check
	./scripts/convert_figures_svg_to_pdf.sh

# Stage a minimal anonymized bundle under tmp/anonymized_bundle/
anonymized-bundle:
	./scripts/package_anonymized_bundle.sh

# Guardrail: ensure all \\cite{...} keys referenced in the main draft exist in references.bib
citations-check:
	./scripts/check_citations_vs_bib.sh

# Build the working EMNLP main paper PDF (review mode, with line numbers)
# Output: docs/paper/latex_paper_emnlp2023/main.pdf
paper:
	cd docs/paper/latex_paper_emnlp2023 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Build a camera-ready style PDF (no line numbers) for page counting
paper-camera-ready:
	cd docs/paper/latex_paper_emnlp2023 && latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-pdflatex='pdflatex %O "\\def\\CAMERAREADY{1}\\input{%S}"' main.tex

# Clean LaTeX build artifacts under the working paper directory
paper-clean:
	cd docs/paper/latex_paper_emnlp2023 && latexmk -C
