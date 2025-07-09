# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Will Handley's academic talks repository, containing LaTeX beamer presentations for conferences and seminars. The repository uses a branch-based system where each talk is developed on its own branch (e.g., `ini_2025`, `sardinia_2025`, `bristol_2025`), with the master branch containing the common template infrastructure.

## Development Commands

### LaTeX Build Commands
- **Build PDF**: `pdflatex will_handley.tex` (run twice for proper references)
- **Build with latexmk**: `latexmk -pdf will_handley.tex` (recommended - handles multiple passes automatically)
- **Clean build files**: `latexmk -C will_handley.tex`

### Python Environment
- **Activate virtual environment**: `source venv/bin/activate`
- **Deactivate**: `deactivate`
- **Install packages**: `pip install <package>` (after activating venv)

The Python environment includes scientific computing packages (numpy, scipy, matplotlib, pandas) and specialized tools for Bayesian inference (anesthetic, jax, blackjax, jaxopt).

## Architecture and Structure

### Main Template Files
- **`will_handley.tex`**: Main LaTeX document template with placeholder content
- **`will_handley_beamer.sty`**: Beamer-specific styling (colors, layout, navigation)
- **`will_handley.sty`**: General LaTeX packages and custom commands
- **`title_page.sty`**: Author information and institutional logos
- **`matplotlibcolors.sty`**: Matplotlib-compatible color definitions and text color commands
- **`codelistings.sty`**: Code syntax highlighting configuration

### Key Features
- **Color System**: Uses matplotlib default colors (C0-C9) for consistency
- **Custom Commands**: 
  - `\arxiv{number}` for arXiv links
  - `\doi{doi}` for DOI links
  - `\email{address}` for email formatting
  - `\tthref{url}` for website links
  - `\student{photo}{name}{detail}` for student photos
- **Institutional Branding**: Automatically includes university and funding body logos
- **Navigation**: Footer with email, website, and slide numbers

### Directory Structure
- **`logos/`**: Institutional and funding body logos (PDF/PNG)
- **`people/`**: Student and collaborator photos (JPG format)
- **`venv/`**: Python virtual environment (gitignored)
- **Build artifacts**: `.aux`, `.log`, `.pdf`, etc. (gitignored)

## Branch Workflow

Each talk is developed on its own branch named after the conference/venue and year (e.g., `ini_2025`, `sardinia_2025`). The master branch contains the common template infrastructure. When creating a new talk:

1. Create new branch from master
2. Modify `will_handley.tex` with talk-specific content
3. Update title, subtitle, and date in the template
4. Replace placeholder content with actual slides
5. Build and test the presentation

## Common Development Patterns

### Adding New Talks
1. Update title, subtitle, and date in `will_handley.tex`
2. Replace `<+Title+>`, `<+subtitle+>`, `<+Date+>` placeholders
3. Replace `<+Frame title+>` and `<+Content+>` placeholders with actual content
4. Build with `latexmk -pdf will_handley.tex`

### Working with Colors
- Use predefined color commands: `\redtext{}`, `\bluetext{}`, `\greentext{}`, etc.
- Access matplotlib colors directly: `\textcolor{C0}{text}`
- Use `\C[color_number]{text}` for concise color application

### Managing Images
- Place student/collaborator photos in `people/` directory
- Use `\student{filename}{name}{detail}` command for student acknowledgments
- Institutional logos are automatically included in title page

## File Management

### Gitignored Files
- All LaTeX build artifacts (`.aux`, `.log`, `.pdf`, `.nav`, `.out`, `.snm`, `.toc`, `.vrb`)
- Python virtual environment (`venv/`)
- Python cache (`__pycache__/`)

### Important Files to Track
- Template files (`.tex`, `.sty`)
- Images (`logos/`, `people/`)
- Configuration files (`.gitignore`)