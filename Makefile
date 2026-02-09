# GALILEO helper Makefile (paper-facing utilities)

.PHONY: figures-pdf figures-check anonymized-bundle

# Preflight for SVG->PDF conversion tooling
figures-check:
	./scripts/check_figure_tooling.sh

# Convert docs/paper/figures/*.svg -> paper_figures/pdf/*.pdf
figures-pdf: figures-check
	./scripts/convert_figures_svg_to_pdf.sh

# Stage a minimal anonymized bundle under tmp/anonymized_bundle/
anonymized-bundle:
	./scripts/package_anonymized_bundle.sh
