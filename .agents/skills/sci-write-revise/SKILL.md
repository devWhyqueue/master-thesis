---
name: sci-write-revise
description: Review the sentence-level writing quality of existing scientific prose (thesis chapters, reports, papers) by editing it in place. Use when asked to review, check, tighten, or clean up writing, strip clutter, fix passive voice or nominalizations, or check terminology and number consistency. Do not use for drafting new text (use scientific-writing), technical review of methods or results, or citation formatting.
---

# Scientific Writing Review

Review existing prose so that every word earns its place. Improve how claims are delivered; never alter scientific content, data, or technical claims.

Always run all five passes, in order, on the whole text the user points to (a file, a section, or a pasted passage). Apply the revisions directly to the files; for a pasted passage, return the revised text.

## Pass 1: Clutter

Strip every sentence to its cleanest components.

**Dead-weight phrases:**

| Cluttered phrase | Replace with |
|------------------|--------------|
| Due to the fact that | Because |
| A majority of | Most |
| Are of the same opinion | Agree |
| Give rise to | Cause |
| Have an effect on | Affect |
| In the event that | If |
| At the present time | Now / Currently |
| In order to | To |
| A number of | Several / Many |
| On the basis of | Based on |
| In light of the fact that | Because / Since |
| It is worth noting that | (delete, state the point) |
| It is important to note that | (delete) |
| It is interesting to note that | (delete) |
| In terms of | (rewrite to be specific) |

**Dead-weight openers, flag for deletion:** "As it is well known..." (replace with a direct citation), "It should be emphasized that...", "It can be regarded that...", "As it has been shown...", "It is noteworthy that...".

**Redundancy:** remove adjectives or adverbs that repeat what the noun or verb already says: "successful solutions" → "solutions", "completely eliminate" → "eliminate", "future plans" → "plans", "currently underway" → "underway".

## Pass 2: Voice and Verbs

**Passive to active:**

1. Spot the pattern: "to be" verb + past participle ("was observed", "were analyzed").
2. Identify the actor.
3. Reconstruct as subject-verb-object.

Example: "The activation of channels is induced by the depletion of stores." → "Depleting stores activates channels."

Follow the document's voice. For single-author work, do not introduce "we"; make the method, model, or data the subject instead ("The loss reweights rare classes").

**Nominalizations, resurrect the verb:**

| Smothered form | Verb |
|----------------|------|
| Provides a review of | Reviews |
| Offers a confirmation of | Confirms |
| Shows a peak | Peaks |
| Obtains an estimate of | Estimates |
| Conducts an assessment of | Assesses |
| Provides a description of | Describes |
| Makes an adjustment to | Adjusts |
| Performs an analysis of | Analyzes |
| Achieves a reduction in | Reduces |

Check every "noun + of" construction for a direct verb.

**Passive is acceptable** when the actor is unknown or irrelevant, when the document's convention for methods sections requires it, or when the object deserves the emphasis. Do not convert mechanically; flag only passives that obscure who or what acts.

## Pass 3: Sentence Architecture

**Buried predicates:** if more than about 12 words separate subject and main verb, restructure.

- Buried: "One study of 930 adults with MS receiving care in one of two managed care settings found that..."
- Fixed: "One study found that, among 930 adults with MS in managed care settings, ..."

**Punctuation:** a colon sets up a list or explanation in place of a wordy opening; a semicolon links closely related independent clauses in place of a transition word.

**Rhythm:** flag paragraphs where all sentences have roughly the same length (±5 words). Short declarative sentences carry emphasis, longer ones explanation.

## Pass 4: Terminology

**One term per concept.** If the methods say "obese group", the results must not say "heavier group". A synonym for a technical term makes the reader wonder whether a new category was introduced.

1. Extract the key terms from the methods: group names, variable names, technique names, abbreviations.
2. Verify the exact same terms appear in results, discussion, tables, and figure captions.
3. Flag every synonym substituted for a defined term.

**Acronyms:** flag non-standard acronyms created for author convenience. Every acronym must be defined at first use in the abstract, in the main text, and in each table or figure caption.

## Pass 5: Numbers and Citations

**Numbers:**

- Does the sample size in the abstract match the tables?
- Do percentages in the text match the raw numbers in the tables?
- Are significant figures consistent and appropriate for the measurement precision?
- Do figures match the corresponding table values?

**Citations:** flag any statistic stated as established fact but cited only through a secondary source (review, textbook). Recommend verifying the primary source.

## Output

After editing, give a brief summary in chat: a few lines on the dominant kinds of change, not a list of every edit.

List separately, with `file:line`, only what an edit cannot resolve without touching content: number mismatches, a term whose intended meaning is unclear, statistics cited through a secondary source, claims that seem wrong.

## Constraints

- **Never alter scientific content.** If a claim or number seems wrong, flag it; do not change it.
- **Respect the document's conventions** for voice, tense, and citation style.
- **Preserve the author's voice.** If a sentence is clear despite breaking a rule, leave it alone.
