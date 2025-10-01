# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Will Handley's academic talks repository, containing LaTeX beamer presentations for conferences and seminars. The repository uses a **branch-based organization** where each talk is developed on its own branch (e.g., `ini_2025`, `sardinia_2025`, `bristol_2025`), with the master branch containing the common template infrastructure.

**Related repository**: `~/projects/ai_presentation/lectures/` contains a simplified version of this template system for teaching lecture series.

## Development Commands

### LaTeX Build Commands
- **Build PDF**: `pdflatex will_handley.tex` (run twice for proper references)
- **Build with latexmk**: `latexmk -pdf will_handley.tex` (recommended - handles multiple passes automatically)
- **Watch and auto-compile**: `latexmk -pdf -pvc will_handley.tex`
- **Clean build files**: `latexmk -C will_handley.tex`

### Python Environment
- **Activate virtual environment**: `source venv/bin/activate`
- The Python environment includes scientific computing packages (numpy, scipy, matplotlib, pandas) and specialized tools for Bayesian inference (anesthetic, jax, blackjax, jaxopt)

## Architecture and Structure

### Main Template Files

The template system is modular, with styling separated into specialized `.sty` files:

- **`will_handley.tex`**: Main LaTeX document template with placeholder content (`<+Title+>`, `<+subtitle+>`, `<+Date+>`, `<+Frame title+>`)
- **`will_handley_beamer.sty`**: Beamer-specific styling
  - Layout configuration (margins, navigation, itemize settings)
  - Footer with email (`wh260@cam.ac.uk`), website (`willhandley.co.uk/talks`), and slide numbers
  - Slide numbering from zero with appendix support
  - `\student{photo}{name}{detail}` command for student acknowledgments (places photo in top-right corner)
- **`will_handley.sty`**: General LaTeX packages and custom citation commands
  - Package imports (amsmath, tikz, pgfplots, siunitx, etc.)
  - Custom citation commands: `\arxiv{number}`, `\doi{doi}`, `\email{address}`, `\tthref{url}`, `\github{user/repo}`, `\xkcd{number}`
  - Date formatting with `datetime2` package
- **`title_page.sty`**: Author information and institutional logos (KICC, Cambridge, IoA, DAMTP, Cavendish)
- **`matplotlibcolors.sty`**: Color definitions and text color commands
  - Matplotlib default colors (C0-C9): `#1f77b4` (blue), `#ff7f0e` (orange), `#2ca02c` (green), `#d62728` (red), etc.
  - Text color commands: `\redtext{}`, `\bluetext{}`, `\orangetext{}`, `\greentext{}`, `\yellowtext{}`, `\blacktext{}`
  - Generic color command: `\C[n]{text}` (defaults to C1 if n omitted)
- **`codelistings.sty`**: Code syntax highlighting configuration

### Directory Structure
- **`logos/`**: Institutional and funding body logos (PDF/PNG)
- **`people/`**: Student and collaborator photos (JPG format) for use with `\student{}` command
- **`figures/`**: Talk-specific figures and diagrams
- **`venv/`**: Python virtual environment (gitignored)
- **Build artifacts**: `.aux`, `.log`, `.pdf`, `.nav`, `.out`, `.snm`, `.toc`, `.vrb` (gitignored)

## Branch Workflow

Each talk is developed on its own branch named after the conference/venue and year (e.g., `ini_2025`, `sardinia_2025`, `bristol_2025`). The master branch contains the common template infrastructure.

### Creating a New Talk

1. **Create branch from master**:
   ```bash
   git checkout -b conference_year
   ```
2. **Modify `will_handley.tex`**:
   - Replace `<+Title+>`, `<+subtitle+>`, `<+Date+>` placeholders
   - Replace `<+Frame title+>` placeholders with actual slide content
3. **Build and iterate**:
   ```bash
   latexmk -pdf -pvc will_handley.tex  # Auto-recompile on save
   ```

### Working with the Template

**Color usage**:
- Predefined commands: `\redtext{}`, `\bluetext{}`, `\orangetext{}`, `\greentext{}`, `\yellowtext{}`, `\blacktext{}`
- Direct matplotlib colors: `\textcolor{C0}{text}`, `\textcolor{C3}{text}`, etc.
- Concise notation: `\C[0]{blue text}`, `\C[3]{red text}`

**Custom citation commands**:
- `\arxiv{2301.07041}` → clickable arXiv link
- `\doi{10.1093/mnras/stab3282}` → clickable DOI link
- `\email{wh260@cam.ac.uk}` → formatted email link
- `\tthref{willhandley.co.uk/talks}` → clickable URL
- `\github{williamjameshandley/anesthetic}` → GitHub link

**Student acknowledgments**:
- Place photos in `people/` directory (JPG format)
- Use `\student{photo.jpg}{Student Name}{Detail}` on a slide
- Photo appears in top-right corner with name and detail overlay

**Beamer blocks and alerts**:
- `\alert{text}` → highlighted text (red/C3)
- `\begin{block}{Title}...\end{block}` → blue block (C0)
- `\begin{exampleblock}{Title}...\end{exampleblock}` → green block (C2)
- `\begin{alertblock}{Title}...\end{alertblock}` → red block (C3)