# Nous: Academic Paper

Paper for arXiv submission (cs.AI): "Nous: Extracting and Injecting Cognitive Priors for Diverse LLM Agents"

## Build Instructions

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# Or minimal install
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-latex-recommended \
  texlive-fonts-recommended texlive-science texlive-bibtex-extra biber
```

Required LaTeX packages: `amsmath`, `amssymb`, `booktabs`, `multirow`, `algorithm2e`, `tikz`, `pgfplots`, `natbib`, `hyperref`, `subcaption`, `xcolor`, `enumitem`, `microtype`, `geometry`, `setspace`, `fancyhdr`, `titlesec`.

### Build

```bash
cd ~/nous/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Three `pdflatex` passes are required to resolve all cross-references and citations.

### Quick rebuild (after edits)

```bash
pdflatex main.tex
```

## File Structure

```
paper/
  main.tex              # Full paper (~14 pages + ~6 pages appendix)
  references.bib        # ~35 BibTeX entries
  figures/
    agent_architecture.tex    # Fig 1: Agent layers with Nous
    core_shell_membrane.tex   # Fig 2: Three-layer concentric model
    extraction_pipeline.tex   # Fig 3: End-to-end pipeline flowchart
    experiment_results.tex    # Fig 4: Differentiation heatmap
  README.md             # This file
```

## arXiv Submission Checklist

- [ ] Build compiles without errors
- [ ] No undefined references (`\ref`, `\cite`)
- [ ] All figures render correctly
- [ ] Page count: ~14 pages main text + ~6 pages appendix
- [ ] Abstract under 300 words
- [ ] Author email valid
- [ ] All `.bib` entries have complete fields
- [ ] No absolute file paths in source
- [ ] Category: `cs.AI` (primary), `cs.HC`, `cs.MA` (secondary)
- [ ] License: CC BY 4.0

### Submission bundle

```bash
# Create submission archive
tar czf nous-paper.tar.gz main.tex references.bib figures/ README.md
```

arXiv accepts `.tar.gz` uploads. Do not include compiled PDFs or auxiliary files (`.aux`, `.bbl`, `.log`).

## Key Source Files Referenced

| Source File | Paper Section |
|---|---|
| `engine/schema/nous_schema.py` | Table 1, Section 3, Appendix A |
| `engine/schema/defaults.py` | Section 4.4 (population priors) |
| `engine/extractors/risk_perception.py` | Algorithm 2, Section 4.3 |
| `engine/extractors/*.py` | Section 4.3 (all extractors) |
| `injection/translator.py` | Section 5.1, Appendix C |
| `injection/injector.py` | Section 5.3 |
| `injection/experiment.py` | Section 6.1 |
| `injection/results.json` | Section 6.2-6.3, Appendix B |
| `injection/results_en.json` | Section 6.4, Appendix B |
