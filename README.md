# translation-lab

Code from a scoped set of LLM dataset-translation experiments. It covers document planning, translation requests, reconstruction, automatic tripwires, COMET-QE packing, and blinded review.

The retained experiment data are published as
[`RfKnowledge/dolci-think-translation-tests`](https://huggingface.co/datasets/RfKnowledge/dolci-think-translation-tests).

## Setup

```bash
uv sync --dev
uv run prek install --hook-type pre-commit --hook-type pre-push
uv run translation-lab --help
```

Model-template helpers and COMET-QE use incompatible Transformers major versions, matching the separate generation and scoring environments used in the experiments:

```bash
uv sync --extra models       # Transformers 5.6 model templates
uv sync --extra comet        # COMET-QE and Transformers 4
```

## Input

Commands expect UTF-8 JSONL with one object per line. Each object needs a unique `id` and a string `src_text` containing one complete document:

```json
{"id":"dolci-think-000001","src_text":"<think>Reasoning to translate...</think>Final answer."}
```

If a dataset contains conversations, first select and flatten the message contents to translate. `--id-field` and `--text-field` change the field names.

## Retained methods

- `P0` recognizes reasoning regions, fenced code, display mathematics, and Markdown tables. Python retains literal structures, translates prose in bounded units and natural-language table cells separately, then reconstructs the document.
- `Pc` uses the same plan and current unit as P0, with up to three preceding source and cleaned translation pairs as chat history. It generates only the current translation.
- `RAW` uses bounded source chunks without structure-aware parsing. Python retains only exact boundary separators.
- `SB` uses the same lossless raw chunks and separators as RAW. It puts the
  translation contract in the system message and encloses each source chunk in
  collision-checked sentinel boundaries in the user message. It does not parse
  or protect code, mathematics, tables, or reasoning regions.
- `A1` and `A2` expose one P0 unit and request one translation as prompt-only JSON or schema-constrained JSON.
- `B1` and `B2` expose a tokenizer-bounded consecutive window but request one designated translation.
- `B3` and `B4` expose the same kind of window and request every translation in it. The even-numbered methods use a JSON Schema through vLLM/XGrammar.

The default P0 label is 512 and its character target is 2,048. The label is experimental metadata, not a claim that each translation unit contains 512 model tokens.

## Run an experiment condition

Start an OpenAI-compatible server, then run one model, method, and target language at a time:

```bash
vllm serve --config configs/vllm/gemma4-curve.yaml

uv run translation-lab run-experiment input.jsonl runs/gemma4-p0-fr \
  --base-url http://127.0.0.1:8000/v1 \
  --language fr \
  --method P0 \
  --model-key gemma4
```

`run.json` fixes the checkpoint, retained revision when available, interface, method, language, chunk settings, generation parameters, and rendering mode. `results.jsonl` stores one atomic evidence record per source. Each record contains:

- the ordered translation units and executable reconstruction recipe;
- every request's visible and requested unit indices, exact messages or rendered prompt, raw response, processed translations, finish reason, token counts, parser result, and unit tripwires;
- the assembled document, completeness and reconstructability flags, failure reasons, and document tripwires.

The result is appended only after that source's requests and evidence validation finish. A method failure is still retained with `original_run_complete=false`. `--resume` skips source IDs already present. Replay every retained reconstruction with:

```bash
uv run translation-lab validate-evidence runs/gemma4-p0-fr
```

Run all six language conditions by invoking the command separately with `pl`, `de`, `fr`, `es`, `fi`, and `el`. This keeps checkpointing and accounting unambiguous.

The published work contains these representative matrices:

| Comparison | Models | Methods | Languages |
| --- | --- | --- | --- |
| Chunk curve | Gemma 3, Gemma 4 | P0 at labels 300, 512, 1024, 2048, 4096, and 8192 | all six |
| Previous translations | Gemma 4 | P0, Pc at label 512 | all six |
| Model and parser | Gemma 4, MiLMMT, TranslateGemma, Hy-MT2 | P0, RAW at label 512 | six, with Hy-MT2's four official languages distinguished from Finnish and Greek |
| New general model | Qwen3.8 27B FP8 | P0 at label 512 | all six |
| JSON output | Gemma 4 | P0 comparator, A1 through B4 at label 512 | all six |
| System boundary | Gemma 4 | SB at labels 300, 512, 1024, 2048, 4096, and 8192 | all six |

For the curve, `chunk_target_chars` was four times the categorical label. The retained prose-output ceilings were 8,192 tokens through label 1024, 12,288 at 2048, 20,480 at 4096, and 32,768 at 8192. `run-experiment` selects those values for Gemma P0 unless `--max-output-tokens` overrides them. Table cells retain their 512-token ceiling. Each later comparison used the same 340-source slice and the P0 label-512 plan where applicable. The Hugging Face dataset card identifies retained conditions and plan revisions. The package does not infer an experiment matrix from filenames.

### Model-native interfaces

`--model-key` selects the retained interfaces: `gemma3`, `gemma4`, `qwen3.8`, `translategemma`, `milmmt`, or `hy-mt2`. P0 and RAW support every interface. Pc and A1-B4 were retained only with Gemma 4. The `configs/vllm` directory includes the retained serving settings. Hy-MT2 also used the Triton unquantized-MoE backend, and Qwen used vLLM 0.25.1 with the direct `deep_gemm` linear backend after the automatic FlashInfer path failed during startup.

For the exact offline-rendered request path used in the model comparison, install the `models` extra and pass the local checkpoint directory:

```bash
uv run --extra models translation-lab run-experiment input.jsonl runs/qwen-p0-fr \
  --base-url http://127.0.0.1:8000/v1 \
  --language fr \
  --method P0 \
  --model-key qwen3.8 \
  --renderer /models/Qwen3.8-27B-FP8
```

The local tokenizer or processor renders the native template once. The completion endpoint receives that prompt with `add_special_tokens=false`. Qwen thinking is disabled and its two retained stop-token IDs are sent explicitly. TranslateGemma always requires `--renderer` and its official 2,048-token input condition is checked before each request and over-budget units are rejected rather than subdivided.

### System-boundary requests

Run SB exactly like another experiment method:

```bash
uv run translation-lab run-experiment input.jsonl runs/gemma4-sb-fr-512 \
  --base-url http://127.0.0.1:8000/v1 \
  --language fr \
  --method SB \
  --model-key gemma4 \
  --chunk-label 512 \
  --chunk-chars 2048
```

For every raw chunk, SB sends a system message containing the complete
translation contract and a user message of this form:

```text
Payload to be translated:
<BEGIN_TRANSLATION_PAYLOAD>
{source chunk}
<END_TRANSLATION_PAYLOAD>
```

The system message says that instructions, questions, problems, examples, and
requested output formats inside the payload are data to translate rather than
commands to follow. The runner rejects a source if either sentinel occurs
exactly or after NFKC case-folding. It also rejects native chat-envelope strings
that could interfere with template boundaries. A returned sentinel is recorded
as a protocol failure and severe alarm.

The retained Gemma 4 SB curve used provider sampling
`temperature=1.0`, `top_p=0.95`, `top_k=64`, seed 42, and stop-token IDs
`[1, 50, 106]`. The retained Qwen SB mechanism test used non-thinking mode,
`temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=1.5`,
`repetition_penalty=1.0`, seed 42, and stop-token IDs `[248046, 248044]`.
Those settings are method-specific; the earlier P0 and RAW records keep their
historical generation parameters.

The SB labels use target character counts of 1,200, 2,048, 4,096, 8,192,
16,384, and 32,768. Its output ceilings are 8,192 tokens through label 1024,
12,288 at 2048, 20,480 at 4096, and 32,768 at 8192. The labels are historical
names, not model-token guarantees.

JSON methods also require `--renderer` so the same windows fit every one of the six rendered language prompts under the 8,192-token prompt budget. With vLLM 0.22.1, schema experiments require the repaired XGrammar integration to receive Gemma 4's declared stop-token IDs `[1, 50, 106]`. A JSON Schema can guarantee syntax and cardinality only when that server integration is correct but it does not guarantee a complete or faithful translation.

The repair is hash-gated to the retained vLLM 0.22.1 and XGrammar 0.2.1 source. Run it inside that serving environment, then keep the validation report with the experiment:

```bash
translation-lab xgrammar-stop-fix \
  --site-packages /path/to/python/site-packages \
  --model /models/gemma-4-31B-it-FP8-Dynamic \
  --report xgrammar-stop-fix.json \
  --apply
```

The command refuses an unknown backend hash rather than applying the edit to a different vLLM version. Prefer an upstream vLLM release containing the same fix for new work.

## Automatic checks

Each unit records output/source length, target and English function-word density, new code fences, new foreign-script characters, repeated 8-grams, suspicious invisible-character runs, and exact literal preservation. Document evidence also records reasoning-tag, fence, wrapper-leak, and preservation checks. Post-hoc `langid` runs on stripped prose only when at least 30 letters remain, its result is diagnostic and does not override the calibrated function-word gate.

Source evidence records the fraction of letters in CJK, Kana, or Hangul and flags the retained 5% source-contamination boundary. The reported quality populations excluded those source-side language confounds. The package records the flag and leaves population selection to the analysis.

The severe composite contains operational alarms for length endings, empty outputs, English reversion, new code fences, runaway expansion, undertranslation, repetition, and invisible or format-character runs. It is a tripwire, not a translation-quality score.

## COMET-QE

The package does not download a checkpoint. Bring a local reference-free checkpoint such as `wmt22-cometkiwi-da` and score a completed condition in the separate COMET environment:

```bash
uv run --extra comet translation-lab comet input.jsonl runs/gemma4-p0-fr/results.jsonl \
  runs/gemma4-p0-fr/comet.json \
  --checkpoint /models/wmt22-cometkiwi-da
```

The scorer strips non-prose blocks, uses the checkpoint's InfoXLM tokenizer, packs each source and candidate pair under the 512-token encoder limit, and records whether each scoring unit used whole, sentence, or positional alignment. Document COMET is weighted by combined source and candidate token length and worst-unit COMET is the minimum unit score. The report retains each unit, its token counts, weight, alignment, and score.

An incomplete document is not assigned an invented COMET penalty. It remains a hard operational failure and is excluded from COMET, with the excluded count reported. Interpret COMET only beside coverage, completeness, alignment modes, and the tripwires.

## Legacy translation command

The smaller original P0-only interface remains available:

```bash
uv run translation-lab translate input.jsonl output.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --language French
```

It writes the assembled output, source/candidate pairs, request counts, finish reasons, and basic tripwires. Use `run-experiment` when request topology and replayable evidence matter.

## Blinded comparison review

Prepare two or three variants, keep the condition map away from the judge, and call a separate OpenAI-compatible judge endpoint:

```bash
uv run translation-lab prepare-review review-sources.jsonl blinded-items.jsonl private-map.jsonl \
  --candidate 'gemma=gemma.jsonl' \
  --candidate 'qwen=qwen.jsonl' \
  --language French \
  --task-type model_comparison

uv run translation-lab review blinded-items.jsonl judge-results.jsonl \
  --model judge-model \
  --base-url http://127.0.0.1:8001/v1

uv run translation-lab validate-review blinded-items.jsonl judge-results.jsonl

uv run translation-lab summarize-review \
  blinded-items.jsonl private-map.jsonl judge-results.jsonl review-report.json \
  --native-review native-review-items.jsonl
```

The rubric is in [rubric.md](src/translation_lab/rubric.md). Model review is triage, not a substitute for native-speaker validation.

## Verify the package

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pyrefly check
uv run pytest
uv run prek run --all-files
```
