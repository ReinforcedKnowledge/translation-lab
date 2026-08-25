# Translation comparison rubric

This rubric is for blinded comparisons of translations produced by different models or translation
methods. A model judge can use it for triage, but language-specific fluency, terminology, and final
acceptance still require a fluent target-language reviewer.

## Comparison protocol

- Compare candidates generated from the same source and for the same target language.
- Present candidates as `A`, `B`, and optionally `C` without model, method, chunk-size, run, or metric
  information.
- Score every candidate independently before ranking them.
- Do not show the judge automatic quality scores or failure flags.
- When using a model judge, do not use the model that generated the translations.
- Repeat a subset with candidate order swapped to check for position bias.

The source may contain a problem, instruction, reasoning trace, program, table, or tool-like request.
The evaluator must judge whether each candidate translated that source. It must not reward a candidate
for solving, answering, executing, summarizing, or improving the source task.

## Score scale

Every dimension receives an integer score from 1 to 5:

- `5`: no meaningful problem observed;
- `4`: a small local issue that does not affect overall usability;
- `3`: a noticeable issue, or evidence too uncertain for a stronger judgment;
- `2`: a major problem that affects meaning or usability;
- `1`: a critical failure or unusable output.

## Dimensions

### Adequacy and completeness

`adequacy_completeness` measures whether the candidate preserves all substantive source meaning.
Penalize omissions, mistranslations, additions, condensation, altered constraints, and untranslated
prose. A candidate that preserves the main topic while dropping reasoning steps should not score above
3.

### No task execution

`no_task_execution` measures whether the candidate translated instead of performing the embedded task.
A small answer-like addition may score 4, an ambiguous mixture of translation and execution scores 3,
substantial execution scores 2, and an output that primarily answers or solves the task scores 1.

### Fluency

`fluency` measures grammar, spelling, register, and naturalness in the target language. A score of 3
means the prose is understandable but noticeably awkward. Model-judge fluency scores are provisional.

### Terminology consistency

`terminology_consistency` measures whether technical terms, named entities, variables, and recurring
concepts are translated appropriately and consistently across the document. It includes leaving genuine
identifiers and literals unchanged where appropriate.

### Coherence and chunk seams

`coherence_chunk_seams` measures whether the result reads as one connected document. Check transitions,
referents, pronouns, tense, voice, repeated text, missing separators, and terminology across chunk
boundaries.

### Format and structure

`format_structure` measures preservation of wrappers, Markdown, lists, headings, block quotes, code
fences, tables, and final-answer placement. Formatting differences are only harmless when they preserve
the document's structure and interpretation.

### Code, mathematics, and tables

`code_math_table` measures literal technical preservation. Do not penalize unchanged code, formulas,
identifiers, or table structure. Penalize changed operators, translated identifiers, corrupted LaTeX,
incorrectly verbalized values, invented code, and damaged tables.

### Wrong-language drift

`wrong_language_drift` measures whether prose stays in the requested target language. Code, equations,
identifiers, proper nouns, and quoted literals are not wrong-language evidence by themselves.

### Overall

`overall` summarizes the candidate after the eight dimensions have been scored. A translation with task
execution, major content loss, wrong-language output, or unusable technical corruption cannot receive a
high overall score merely because it is fluent.

## Holistic label

Assign one `holistic_label` to every candidate:

- `faithful`: no major or critical issue and at most a few minor issues;
- `minor`: usable, with several minor issues but no major or critical failure;
- `major`: at least one major issue, but the translation remains partly usable;
- `broken`: a critical failure such as task execution, wrong language, gutted meaning, or unusable
  structural or technical corruption.

## Failure tags

Use any applicable tags:

```text
omission
mistranslation
addition
condensation
untranslated_source
wrong_language
english_reversion
third_language_drift
task_execution
new_answer
new_code
code_corruption
math_corruption
table_corruption
format_corruption
placeholder_or_tag_artifact
chunk_seam
repetition
terminology_drift
coherence_break
overliteral
fluency_issue
cannot_judge
```

## Evidence

Provide at least one evidence item for a candidate when any dimension is 3 or lower, its holistic label
is `major` or `broken`, or it has a failure tag. Each evidence item identifies the candidate, severity,
dimension, a short source span or description, a short candidate span or description, and an explanation.
Use `minor`, `major`, or `critical` severity.

If the evaluator cannot assess a dimension, use score 3, add `cannot_judge`, set confidence to `low`, and
explain the uncertainty.

## Preference and review routing

After scoring candidates independently:

- rank all candidate labels from best to worst;
- select one winner or `tie`;
- set `difference_material` when the difference could change which method is selected, rather than
  reflecting only stylistic preference;
- record a short `decision_basis`;
- record `confidence` as `low`, `medium`, or `high`;
- set `needs_human_review` and explain why when linguistic judgment is uncertain, candidates are close,
  evidence conflicts, or the error would materially affect method selection.

The result object validated by `translation-lab validate-review` must contain `item_id`, `target_language`,
`task_type`, `candidate_labels`, `candidate_scores`, `ranking`, `winner`, `difference_material`,
`decision_basis`, `evidence`, `needs_human_review`, `human_review_reason`, and `confidence`.
