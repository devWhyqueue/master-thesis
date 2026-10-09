---
name: scientific-writing
description: Draft, revise, and structure clear, precise scientific writing for papers, theses, LaTeX documents, literature reviews, methods, results, discussions, captions, tables, equations, and figures. Use whenever working on academic prose, scientific arguments, or research visuals.
---

# Scientific Writing

Produce scientific writing that is **easy to understand without sacrificing precision, nuance, or rigor**. Optimize for reader comprehension, not perceived sophistication. Every sentence should contribute to the reader's understanding.

## Workflow

1. Identify the target artifact, intended audience, and central message.
2. Inspect surrounding files before editing to match existing structure, terminology, macros, citations, and tone.
3. Preserve the intended claims and scientific contribution. Flag missing evidence, ambiguities, and citation gaps; never invent support.
4. Draft complete, coherent prose. Avoid placeholders, outline stubs, and headings without substance.
5. Use figures, tables, or equations where they communicate an idea more clearly than prose.
6. **Review explicitly for readability**, then verify scientific accuracy, LaTeX structure, and layout.

## Readability First

- **Prefer simple, direct language.** Use familiar words, concrete nouns, and active constructions where appropriate. Avoid academic-sounding vocabulary that adds no precision.
- **Make sentences easy to follow.** Prefer short, logically connected sentences. Split sentences with multiple independent claims, excessive clauses, or complicated dependencies.
- **Make the reasoning explicit.** Explain what a concept means, why it matters, and how it connects to the argument. Do not force readers to infer missing logical steps.
- **Explain intuition before formalism.** Introduce the underlying idea before detailed mechanisms, notation, or equations when possible.
- **Use technical terminology deliberately.** Retain established terms when they add precision, define unfamiliar ones on first use, and avoid unnecessary jargon.
- **Keep paragraphs focused.** Develop one main idea per paragraph with a clear claim, supporting evidence or reasoning, and a connection to the broader argument.
- **Remove unnecessary complexity.** Eliminate filler, repetition, inflated phrasing, redundant qualifiers, and abstract descriptions that could be stated concretely.
- **Maintain information fidelity.** Simplification must never change scientific meaning, remove important qualifications, or conceal uncertainty.

Before finalizing substantial prose, perform a clarity pass: identify sentences that require rereading, simplify their structure, remove unnecessary terminology, and check that all essential meaning remains intact.

## Scientific Argumentation

- Distinguish observations, assumptions, hypotheses, interpretations, and established findings.
- Make claims specific and appropriately supported. Avoid hype and unsupported novelty claims.
- Explain representative related work and its relevance rather than listing citations or creating citation piles.
- For datasets, explain their relevance and limitations, not just their characteristics.
- In methods, distinguish what is varied, what remains fixed, and what each design choice is intended to test.
- In results, prioritize the main finding, supporting evidence, and interpretation over implementation details.
- Keep abstracts focused on the problem, approach, principal findings, and implications.
- Use reader-facing scientific terminology instead of internal project names, code references, tuning jargon, run counts, file paths, pipeline stages, hardware details, or runtime logs unless essential.
- Make transitions between motivation, prior work, methods, results, limitations, and implications explicit.
- Match the author's voice. For single-author work, avoid editorial "we"; prefer direct or impersonal phrasing without overusing the passive voice.
- Match existing citation conventions. Use natural author-year citations when applicable; for numeric citations, avoid excessive sentence-by-sentence references unless necessary.
- Treat revisions as standalone final prose, not change logs. Report the experiment as it finally stands; never narrate amendments, protocol deviations, or reruns.

## Structure and LaTeX

- Keep one root document responsible for the preamble and single PDF build target. Place larger chapters or units in separate `.tex` files included through `\input{...}`.
- Keep section hierarchy restrained. Create subsections only for substantial, distinct topics; merge short or closely related blocks.
- Never place headings directly next to other headings without meaningful orienting prose.
- Use consistent labels and cross-references for sections, figures, tables, and equations.
- Define mathematical symbols near first use. Introduce equations only when they clarify a concept, definition, objective, or method.
- Use `\emph{...}` when first introducing specialized terms, if appropriate to the document's style.
- Format fractional values with decimal points and thousands with commas (e.g., `0.72`, `2,152`). Prefer `\num{...}` when the document uses `siunitx`.
- Maintain reader orientation in longer sections through concise signposting rather than excessive headings.

## Figures and Tables

- Prefer figures, diagrams, tables, and equations when they improve comprehension.
- Use TikZ for simple conceptual diagrams; use other suitable visualization tools for more complex figures.
- Make visuals understandable at paper scale, with legible labels, no overlapping curves or hidden markers, at most two panels per row, and dimensions within `\textwidth`.
- Include a panel or table only if the text draws a claim from it. Keep legends, colours, and units consistent across panels and datasets.
- Use tables for comparisons, taxonomies, ablations, and structured summaries.
- Wrap tables in `table` floats with `\caption{...}` and `\label{tab:...}`. Follow existing caption placement; otherwise place captions below tables.
- Keep included table fragments limited to the tabular body; define captions and labels in the parent document.
- Give figures and tables informative captions explaining their main takeaway, not merely their contents.
- When prose directly continues after a display or list, use `\noindent` where needed to avoid unintended indentation.

## Final Review

Before finishing, verify:

1. **Clarity:** Can an informed reader understand each paragraph on first reading? Are the main ideas, logical connections, and terminology clear?
2. **Precision:** Are all claims supported, qualifications preserved, and symbols defined?
3. **Structure:** Does each section advance the argument without fragmentation, repetition, or unnecessary detail?
4. **Visuals:** Are figures, tables, captions, labels, and mathematical notation correct and readable?
5. **LaTeX:** Are included files body-only, references consistent, and layouts free of overlaps or overfull boxes?

Compile or run the existing LaTeX checks when feasible. Otherwise report that compilation was not performed.
