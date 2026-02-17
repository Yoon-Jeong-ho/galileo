# GALILEO helper Makefile (paper-facing utilities)

.PHONY: figures-pdf figures-check anonymized-bundle citations-check

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
