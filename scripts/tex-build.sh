#!/usr/bin/env bash
# Compile a LaTeX report: pdflatex, bibtex, pdflatex x2 in a temporary build dir.
# Only the final PDF is copied next to the .tex; aux files never touch the worktree.
# Usage (Git Bash): scripts/tex-build.sh path/to/report.tex
# Prints overfull/underfull boxes and undefined references/citations from the final pass.
# On failure, .latex-build is kept next to the .tex for the log; delete it once fixed.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: tex-build.sh path/to/report.tex" >&2; exit 2; }
bin=/c/texlive/2026/bin/windows
name=$(basename "$1" .tex)
build=.latex-build
cd "$(dirname "$1")"

run() { "$@" > /dev/null || { echo "$1 failed, see $PWD/$build/$name.log" >&2; exit 1; }; }
latex() { run "$bin/pdflatex.exe" -interaction=nonstopmode -halt-on-error "-output-directory=$build" "$name.tex"; }

mkdir -p "$build"
shopt -s nullglob
bibs=(*.bib)
latex
if [ ${#bibs[@]} -gt 0 ]; then
  cp "${bibs[@]}" "$build/"
  (cd "$build" && run "$bin/bibtex.exe" "$name")
fi
latex
latex

grep -E '^(Overfull|Underfull) \\[hv]box|undefined' "$build/$name.log" \
  || echo "clean: no box or undefined-reference warnings"
cp "$build/$name.pdf" "$name.pdf"
rm -rf "$build"
