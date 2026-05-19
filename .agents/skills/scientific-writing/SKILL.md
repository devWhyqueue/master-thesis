---
name: scientific-writing
description: Scientific writing guidance for drafting, revising, structuring, and polishing research papers, thesis chapters, LaTeX documents, literature review sections, methods/results/discussion text, captions, tables, equations, and conceptual figures. Use when Codex works on academic prose, paper or thesis organization, LaTeX sectioning, scientific argument flow, or research visuals for papers and theses.
---

# Scientific Writing

Use this skill to produce scientific writing that is clear, structured, evidence-aware, and ready to live in a paper or thesis source tree.

## Workflow

1. Identify the target artifact: paper section, thesis chapter, abstract, related work, methods, results, discussion, caption, table, equation, or figure.
2. Inspect surrounding files before editing existing LaTeX so the sectioning, macros, labels, citation style, and tone match the document.
3. Preserve the document's thesis, claims, and citation obligations. Flag missing evidence, unclear claims, or citation gaps instead of inventing support.
4. Draft in complete scientific prose. Avoid placeholder-only paragraphs, outline stubs, and headings without substance.
5. Add or recommend figures, tables, equations, or schematic diagrams when they make the argument more concrete.
6. Check LaTeX structure and visual layout before finishing.

## LaTeX Structure

- Compile only one PDF output per LaTeX document.
- Keep the root document responsible for the preamble and build target. Put larger chapter or unit bodies in separate `.tex` files and include them from `main.tex` with `\input{...}`.
- Do not give included files their own document preamble. Do not compile included chapter or unit files into separate PDFs.
- Keep section hierarchy restrained. Use top-level sections only for real chapters or major document units.
- Use `\subsection{...}` for meaningful internal blocks, but keep each subsection substantial enough to preserve reading flow.
- Do not create a new subsection after only one short paragraph. Combine closely related points into a larger subsection.
- Never place `\section{...}` directly before `\subsection{...}`, or one `\subsection{...}` directly before another heading. Add meaningful orienting prose between headings, or remove or merge one heading.
- Use labels and references consistently for sections, figures, tables, and equations when the surrounding document does so.

## Writing Standards

- Prefer precise claims over broad declarations. State what is known, what is assumed, and what follows from the cited evidence.
- Build paragraphs around one main point: topic sentence, supporting evidence or reasoning, and a clear connection to the section's purpose.
- Use reader-facing scientific terms instead of local project, pipeline, file-structure, hardware, or runtime-log language. Define specialized terms on first use, using `\emph{...}` for the introduced term when writing LaTeX, then continue in plain text.
- Keep abstracts focused on the problem, scope, key result, and interpretation; omit low-level implementation details unless they are essential to the claim.
- Match the document's existing citation style. For author-year documents, use parenthetical or narrative author-year citations naturally, e.g., `(Goodfellow et al., 2016)` or `Goodfellow et al. (2016)`. For numeric styles, place citations outside sentence grammar at the end of the relevant paragraph, not after each sentence, unless a local convention clearly requires otherwise.
- Avoid citation piles. For important method families, concepts, or debates, briefly explain representative work and connect it to the current study.
- For datasets, materials, or corpora, explain why they fit the research question and what limitations or biases matter, not only their basic counts or properties.
- In methods prose, distinguish what is varied, what is held constant, and what each design choice is meant to isolate.
- Maintain academic restraint. Avoid hype, marketing language, and unsupported novelty claims.
- Make transitions explicit when moving between motivation, prior work, method, result, limitation, and implication.
- Preserve reader orientation in long sections with short signposting paragraphs rather than excessive headings.
- Treat each revision as a standalone final artifact, not as a change log. Avoid meta-revision language such as "revised", "earlier", "previous", "new", "old", or "now" unless the document explicitly needs historical comparison. Describe the current claim, evidence, and interpretation directly.
- When revising, improve argument flow, specificity, and citation placement without changing the author's intended contribution.

## Visuals, Tables, And Math

- Use figures, schematic diagrams, tables, and equations when they clarify concepts more efficiently than prose alone.
- Use TikZ for simple conceptual diagrams such as pipelines, ambiguity examples, task taxonomies, and evaluation flows.
- For more complex or richer visualizations, use the `notebooklm` skill when available and appropriate.
- Keep diagram labels clear and separated from boxes and arrows. Check for overlapping labels before finishing.
- Keep figures compact enough to fit within `\textwidth` without overfull boxes.
- Keep figures readable at paper scale; prefer visual summaries that reveal the main pattern over plots that expose every raw detail.
- Use tables for comparisons, taxonomies, ablations, terminology mappings, and practical checklists.
- Include formulas when they clarify a definition, objective, metric, loss, calibration measure, or evaluation protocol.
- Give every figure and table a short explanatory caption that states what the reader should learn from it.
- When prose resumes immediately after a displayed formula, figure, table, or list environment, use `\noindent` so the continuation does not start with a paragraph indent.

## LaTeX Editing Checklist

- Ensure included `.tex` files contain body content only.
- Ensure headings are separated by useful prose.
- Ensure every new figure or table has a caption and label when the document convention expects labels.
- Ensure formulas define symbols close to first use.
- Ensure any TikZ diagram fits the page, avoids overlapping text, and has readable labels.
- Ensure prose after displays or lists uses `\noindent` when it continues the same discussion.
- Compile or run the document's existing LaTeX check command when feasible; otherwise report that compilation was not run.
