import argparse
import asyncio
import importlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from translation_lab.client import create_generator, create_reviewer, create_translator
from translation_lab.comet import (
    ScoringUnit,
    aggregate_scores,
    load_comet,
    units_from_documents,
    units_from_pairs,
)
from translation_lab.contract import validate_run
from translation_lab.evidence import validate_result
from translation_lab.experiment import (
    DocumentEvidence,
    ExperimentResult,
    RequestEvidence,
    run_document,
)
from translation_lab.interfaces import (
    MODEL_SPECS,
    generation_parameters,
    system_boundary_generation_parameters,
)
from translation_lab.io import append_jsonl, jsonl_ids, read_jsonl, write_jsonl
from translation_lab.metrics import audit_translation, canonical_language
from translation_lab.models import TranslationConfig
from translation_lab.review import (
    build_review_report,
    prepare_review_items,
    review_item,
    rubric_sha256,
    validate_judge_batch,
)
from translation_lab.runner import translate_document
from translation_lab.xgrammar import patch_vllm_0221, validate_stop_fix


def _experiment_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run-experiment", help="run one retained translation method and preserve its evidence"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--language", choices=["pl", "de", "fr", "es", "fi", "el"], required=True)
    parser.add_argument(
        "--method",
        choices=["P0", "Pc", "RAW", "SB", "A1", "A2", "B1", "B2", "B3", "B4"],
        required=True,
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), default="gemma4")
    parser.add_argument("--model")
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="src_text")
    parser.add_argument("--chunk-label", type=int, default=512)
    parser.add_argument("--chunk-chars", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--generation-index", type=int, default=0)
    parser.add_argument("--request-concurrency", type=int, default=16)
    parser.add_argument("--document-concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(handler=_handle_experiment)


def _evidence_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "validate-evidence", help="replay reconstruction from a run-experiment directory"
    )
    parser.add_argument("directory", type=Path)
    parser.set_defaults(handler=_handle_evidence)


def _xgrammar_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "xgrammar-stop-fix", help="apply and validate the vLLM 0.22.1 stop-token repair"
    )
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.set_defaults(handler=_handle_xgrammar)


def _translation_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("translate", help="translate a JSONL dataset")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--language-code")
    parser.add_argument("--model")
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="src_text")
    parser.add_argument("--chunk-chars", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--table-max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(handler=_handle_translate)


def _audit_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("audit", help="audit source/output rows in JSONL")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-field", default="src_text")
    parser.add_argument("--output-field", default="output")
    parser.set_defaults(handler=_handle_audit)


def _comet_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("comet", help="score translations with reference-free COMET")
    parser.add_argument("sources", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--source-field", default="src_text")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(handler=_handle_comet)


def _review_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("review", help="run blinded rubric review")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(handler=_handle_review)


def _review_preparation_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("prepare-review", help="join and blind candidate translations")
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--language", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--source-field", default="src_text")
    parser.add_argument("--output-field", default="output")
    parser.set_defaults(handler=_handle_review_preparation)


def _review_validation_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("validate-review", help="validate blinded judge results")
    parser.add_argument("expected", type=Path)
    parser.add_argument("results", type=Path)
    parser.set_defaults(handler=_handle_review_validation)


def _review_summary_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "summarize-review", help="unblind and summarize validated rubric results"
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--native-review", type=Path)
    parser.add_argument("--force-language", action="append", default=[])
    parser.set_defaults(handler=_handle_review_summary)


def _contract_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("validate-run", help="apply the accepted-run contract")
    parser.add_argument("sources", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--id-field", default="id")
    parser.set_defaults(handler=_handle_contract)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _translation_parser(subparsers)
    _experiment_parser(subparsers)
    _audit_parser(subparsers)
    _comet_parser(subparsers)
    _review_preparation_parser(subparsers)
    _review_parser(subparsers)
    _review_validation_parser(subparsers)
    _review_summary_parser(subparsers)
    _contract_parser(subparsers)
    _evidence_parser(subparsers)
    _xgrammar_parser(subparsers)
    return parser


async def _translate_rows(args: argparse.Namespace) -> int:
    session_started = time.monotonic()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output paths must differ")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; use --resume to append missing rows")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = jsonl_ids(args.output, args.id_field) if args.output.exists() else set()
    translator = await create_translator(
        base_url=args.base_url,
        language=args.language,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_concurrency=args.concurrency,
    )
    config = TranslationConfig(
        language=args.language,
        language_code=args.language_code or canonical_language(args.language),
        chunk_chars=args.chunk_chars,
        max_tokens=args.max_tokens,
        table_max_tokens=args.table_max_tokens,
    )
    rows = (row for row in read_jsonl(args.input) if row.get(args.id_field) not in completed)

    async def translate_row(row: dict[str, Any]) -> dict[str, object]:
        row_id = row.get(args.id_field)
        source = row.get(args.text_field)
        if row_id is None:
            return {"error": f"missing id field {args.id_field!r}"}
        if not isinstance(source, str):
            return {args.id_field: row_id, "error": f"missing text field {args.text_field!r}"}
        started = time.monotonic()
        try:
            result = await translate_document(source, translator, config)
        except Exception as error:
            return {
                args.id_field: row_id,
                "wall_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(error).__name__}: {error}",
            }
        return {
            args.id_field: row_id,
            "wall_seconds": round(time.monotonic() - started, 3),
            **result.to_dict(),
        }

    iterator = iter(rows)
    pending: set[asyncio.Task[dict[str, object]]] = set()
    exhausted = False
    written = 0
    while pending or not exhausted:
        while len(pending) < args.concurrency and not exhausted:
            try:
                row = next(iterator)
            except StopIteration:
                exhausted = True
            else:
                pending.add(asyncio.create_task(translate_row(row)))
        if not pending:
            continue
        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            append_jsonl(args.output, task.result())
            written += 1
    elapsed = time.monotonic() - session_started
    metadata_path = args.output.with_name(args.output.name + ".run.json")
    metadata: dict[str, object] = {"sessions": []}
    if metadata_path.exists():
        loaded = json.loads(metadata_path.read_text())
        if isinstance(loaded, dict) and isinstance(loaded.get("sessions"), list):
            metadata = loaded
    session_values = metadata["sessions"]
    if not isinstance(session_values, list):
        raise TypeError("run metadata sessions must be an array")
    sessions = cast(list[dict[str, object]], session_values)
    metadata["sessions"] = sessions
    sessions.append(
        {
            "elapsed_seconds": round(elapsed, 6),
            "rows_written": written,
            "rows_skipped": len(completed),
            "language": config.language,
            "language_code": config.language_code,
            "model": args.model,
            "base_url": args.base_url,
            "chunk_chars": config.chunk_chars,
            "max_tokens": config.max_tokens,
            "table_max_tokens": config.table_max_tokens,
            "concurrency": args.concurrency,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {written} row(s) to {args.output}")
    return 0


def _handle_translate(args: argparse.Namespace) -> int:
    return asyncio.run(_translate_rows(args))


def _load_renderer(path: Path, model_key: str) -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError("install model helpers with `uv sync --extra models`") from error
    loader = (
        transformers.AutoProcessor if model_key == "translategemma" else transformers.AutoTokenizer
    )
    return loader.from_pretrained(path, local_files_only=True, trust_remote_code=True)


def _token_counter(renderer: Any) -> Callable[[list[dict[str, str]]], int]:
    def count(messages: list[dict[str, str]]) -> int:
        values = renderer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(values, dict) or hasattr(values, "keys"):
            values = values["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            if len(values) != 1:
                raise ValueError("unexpected batched template tokenization")
            values = values[0]
        return len(values)

    return count


async def _experiment_rows(args: argparse.Namespace) -> int:
    if args.request_concurrency < 1 or args.document_concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise ValueError("max output tokens must be positive")
    spec = MODEL_SPECS[args.model_key]
    if args.method in {"Pc", "A1", "A2", "B1", "B2", "B3", "B4"} and args.model_key != "gemma4":
        raise ValueError(f"{args.method} was retained only with the Gemma 4 interface")
    if args.method == "SB" and args.model_key not in {"gemma4", "qwen3.8"}:
        raise ValueError("SB was retained only with Gemma 4 and Qwen3.8")
    if args.method in {"A1", "A2", "B1", "B2", "B3", "B4"} and args.renderer is None:
        raise ValueError("JSON methods require --renderer for exact tokenizer-aware windows")
    if args.model_key == "translategemma" and args.renderer is None:
        raise ValueError("TranslateGemma requires --renderer pointing to its checkpoint")
    renderer = _load_renderer(args.renderer, args.model_key) if args.renderer else None
    token_count = _token_counter(renderer) if renderer is not None else None

    output = args.output_dir / "results.jsonl"
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"{args.output_dir} exists; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained_rows = list(read_jsonl(output)) if output.exists() else []
    retained_ids = [row.get("source_id") for row in retained_rows]
    completed = {str(value) for value in retained_ids if value is not None}
    if None in retained_ids or len(completed) != len(retained_rows):
        raise ValueError("retained results contain a missing or duplicate source_id")
    for row in retained_rows:
        validate_result(_result_from_dict(row))

    parameters = (
        system_boundary_generation_parameters(args.model_key)
        if args.method == "SB"
        else generation_parameters(spec)
    )
    extra_body = cast(dict[str, Any], parameters.pop("extra_body"))
    temperature = float(parameters.pop("temperature"))
    seed = int(parameters.pop("seed"))
    parameters.pop("max_tokens")
    curve_limits = {300: 8192, 512: 8192, 1024: 8192, 2048: 12288, 4096: 20480, 8192: 32768}
    max_output_tokens = args.max_output_tokens
    if max_output_tokens is None:
        if args.method in {"A1", "A2", "B1", "B2", "B3", "B4"}:
            max_output_tokens = 16384
        elif args.method in {"P0", "SB"} and args.model_key in {"gemma3", "gemma4"}:
            max_output_tokens = curve_limits.get(args.chunk_label, spec.max_output_tokens)
        else:
            max_output_tokens = spec.max_output_tokens
    if renderer is None or args.method not in {"P0", "RAW"}:
        extra_body.pop("add_special_tokens", None)
        if args.model_key == "qwen3.8":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    run_contract = {
        "model_key": args.model_key,
        "checkpoint": args.model or spec.checkpoint,
        "checkpoint_revision": spec.revision,
        "interface": spec.interface,
        "method": args.method,
        "target_language": args.language,
        "chunk_label": args.chunk_label,
        "chunk_target_chars": args.chunk_chars,
        "max_output_tokens": max_output_tokens,
        "generation_index": args.generation_index,
        "request_parameters": {
            "temperature": temperature,
            "seed": seed,
            **parameters,
            "extra_body": extra_body,
        },
        "renderer_mode": (
            "request"
            if renderer is not None and args.method in {"P0", "RAW"}
            else "token_count_only"
            if renderer is not None
            else "server_chat_template"
        ),
    }
    run_path = args.output_dir / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != run_contract:
            raise ValueError("--resume contract differs from the retained run.json")
    else:
        run_path.write_text(json.dumps(run_contract, ensure_ascii=False, indent=2) + "\n")
    generator = await create_generator(
        base_url=args.base_url,
        model=args.model or spec.checkpoint,
        api_key=args.api_key,
        timeout=args.timeout,
        max_concurrency=args.request_concurrency,
        temperature=temperature,
        seed=seed,
        extra_body=extra_body,
        generation_options=parameters,
    )

    input_rows = read_jsonl(args.input)
    raw_input_ids = [row.get(args.id_field) for row in input_rows]
    if None in raw_input_ids:
        raise ValueError("input contains a missing source ID")
    input_ids = [str(value) for value in raw_input_ids]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("input contains a missing or duplicate source ID")
    rows = [row for row in input_rows if str(row.get(args.id_field)) not in completed]
    iterator = iter(rows)
    pending: set[asyncio.Task[ExperimentResult]] = set()
    exhausted = False
    written = 0

    async def run(row: dict[str, Any]) -> ExperimentResult:
        source_id = row.get(args.id_field)
        source_text = row.get(args.text_field)
        if source_id is None or not isinstance(source_text, str):
            raise ValueError(f"every row needs {args.id_field!r} and string {args.text_field!r}")
        return await run_document(
            str(source_id),
            source_text,
            args.model or spec.checkpoint,
            args.method,
            args.language,
            generator,
            token_count=cast(Any, token_count),
            chunk_label=args.chunk_label,
            chunk_target_chars=args.chunk_chars,
            generation_index=args.generation_index,
            model_spec=(spec if args.method in {"P0", "RAW"} else None),
            renderer=renderer,
            max_output_tokens=max_output_tokens,
        )

    while pending or not exhausted:
        while len(pending) < args.document_concurrency and not exhausted:
            try:
                row = next(iterator)
            except StopIteration:
                exhausted = True
            else:
                pending.add(asyncio.create_task(run(row)))
        if not pending:
            continue
        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            result = task.result()
            validate_result(result)
            append_jsonl(output, result.to_dict())
            written += 1
    print(f"wrote {written} condition(s) to {output}")
    return 0


def _handle_experiment(args: argparse.Namespace) -> int:
    return asyncio.run(_experiment_rows(args))


def _result_from_dict(row: dict[str, Any]) -> ExperimentResult:
    return ExperimentResult(
        plan=row["plan"],
        requests=[RequestEvidence(**request) for request in row["requests"]],
        document=DocumentEvidence(**row["document"]),
    )


def _handle_evidence(args: argparse.Namespace) -> int:
    path = args.directory / "results.jsonl"
    reports = [validate_result(_result_from_dict(row)) for row in read_jsonl(path)]
    summary = {
        "valid": True,
        "conditions": len(reports),
        "complete": sum(report["original_run_complete"] for report in reports),
        "reconstructable": sum(report["reconstructable"] for report in reports),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _handle_xgrammar(args: argparse.Namespace) -> int:
    application = patch_vllm_0221(args.site_packages) if args.apply else None
    validation = validate_stop_fix(args.site_packages, args.model)
    report = {"application": application, "validation": validation}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _handle_audit(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    count = 0
    for row in read_jsonl(args.input):
        source = row.get(args.source_field)
        candidate = row.get(args.output_field)
        if not isinstance(source, str) or not isinstance(candidate, str):
            raise ValueError("each row must contain string source and output fields")
        append_jsonl(args.output, {**row, **audit_translation(source, candidate, args.language)})
        count += 1
    print(f"audited {count} row(s)")
    return 0


def _dry_comet() -> tuple[Callable[[list[dict[str, str]]], list[float]], Callable[[str], int]]:
    def count_tokens(text: str) -> int:
        return len(text.split()) if text else 0

    def predict(rows: list[dict[str, str]]) -> list[float]:
        scores: list[float] = []
        for row in rows:
            source = max(1, count_tokens(row["src"]))
            candidate = max(1, count_tokens(row["mt"]))
            scores.append(0.5 + 0.4 * min(source, candidate) / max(source, candidate))
        return scores

    return predict, count_tokens


def _handle_comet(args: argparse.Namespace) -> int:
    sources = {
        str(row[args.id_field]): row[args.source_field]
        for row in read_jsonl(args.sources)
        if args.id_field in row and isinstance(row.get(args.source_field), str)
    }
    if args.dry_run:
        predict, count_tokens = _dry_comet()
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --dry-run is used")
        predict, count_tokens = load_comet(args.checkpoint, args.gpus)

    all_units: list[ScoringUnit] = []
    owners: list[object] = []
    alignment = Counter[str]()
    provenance = Counter[str]()
    skipped_incomplete = 0
    for row in read_jsonl(args.results):
        row_id = str(row.get(args.id_field, row.get("source_id")))
        source = sources.get(row_id)
        if not isinstance(source, str):
            continue
        pairs = row.get("pairs")
        if isinstance(row.get("plan"), dict) and isinstance(row.get("requests"), list):
            document = row.get("document")
            if not isinstance(document, dict) or document.get("original_run_complete") is not True:
                skipped_incomplete += 1
                continue
            plan = row["plan"]
            units = {unit["unit_index"]: unit for unit in plan.get("translation_units", [])}
            returned: dict[int, str] = {}
            for request in row["requests"]:
                if request.get("parse_error") is not None:
                    continue
                returned.update(
                    zip(
                        request.get("requested_unit_indices", []),
                        request.get("translations", []),
                        strict=True,
                    )
                )
            if len(returned) != len(units):
                skipped_incomplete += 1
                continue
            if plan.get("method") == "RAW":
                pairs = None
            else:
                pairs = [
                    [unit["source_text"], returned[index]]
                    for index, unit in units.items()
                    if unit.get("kind") == "prose"
                ]
        if isinstance(pairs, list) and pairs:
            normalized = [
                (str(pair[0]), str(pair[1]))
                for pair in pairs
                if isinstance(pair, list) and len(pair) == 2
            ]
            units = units_from_pairs(normalized, count_tokens)
        elif isinstance(row.get("output"), str):
            units = units_from_documents(source, row["output"], count_tokens)
        elif isinstance(row.get("document"), dict) and isinstance(
            row["document"].get("assembled_translation"), str
        ):
            units = units_from_documents(
                source, row["document"]["assembled_translation"], count_tokens
            )
        else:
            continue
        for unit in units:
            if not unit.fits:
                raise RuntimeError(f"over-budget COMET unit for {row_id}")
            if not unit.source.strip() or not unit.candidate.strip():
                continue
            all_units.append(unit)
            owners.append(row_id)
            alignment[unit.alignment] += 1
            provenance[unit.provenance] += 1

    scores = predict([{"src": unit.source, "mt": unit.candidate} for unit in all_units])
    grouped_units: dict[object, list[ScoringUnit]] = defaultdict(list)
    grouped_scores: dict[object, list[float]] = defaultdict(list)
    for owner, unit, score in zip(owners, all_units, scores, strict=True):
        grouped_units[owner].append(unit)
        grouped_scores[owner].append(score)
    documents: dict[str, dict[str, Any]] = {}
    for owner, values in grouped_scores.items():
        mean, minimum = aggregate_scores(grouped_units[owner], values)
        documents[str(owner)] = {
            "score": round(mean, 6),
            "min_unit": round(minimum, 6),
            "units": len(values),
            "evaluation_units": [
                {
                    "source": unit.source,
                    "candidate": unit.candidate,
                    "source_tokens": unit.source_tokens,
                    "candidate_tokens": unit.candidate_tokens,
                    "weight": unit.weight,
                    "alignment": unit.alignment,
                    "provenance": unit.provenance,
                    "score": round(score, 6),
                }
                for unit, score in zip(grouped_units[owner], values, strict=True)
            ],
        }
    means = [float(value["score"]) for value in documents.values()]
    report = {
        "dry_run": args.dry_run,
        "documents": documents,
        "summary": {
            "documents": len(documents),
            "units": len(all_units),
            "mean": round(statistics.mean(means), 6) if means else None,
            "median": round(statistics.median(means), 6) if means else None,
            "alignment": dict(alignment),
            "provenance": dict(provenance),
            "incomplete_documents_with_partial_or_no_units": skipped_incomplete,
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


async def _review_rows(args: argparse.Namespace) -> int:
    session_started = time.monotonic()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output paths must differ")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; use --resume to append missing items")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = jsonl_ids(args.output, "item_id") if args.output.exists() else set()
    reviewer = create_reviewer(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_concurrency=args.concurrency,
    )
    rows = (row for row in read_jsonl(args.input) if row.get("item_id") not in completed)

    async def review_row(
        row: dict[str, Any],
    ) -> tuple[dict[str, object] | None, str | None]:
        item_id = row.get("item_id", "<missing>")
        try:
            result = await review_item(row, reviewer)
        except Exception as error:
            return None, f"{item_id}: {type(error).__name__}: {error}"
        return cast(dict[str, object], result), None

    iterator = iter(rows)
    pending: set[asyncio.Task[tuple[dict[str, object] | None, str | None]]] = set()
    exhausted = False
    written = 0
    failed = 0
    while pending or not exhausted:
        while len(pending) < args.concurrency and not exhausted:
            try:
                row = next(iterator)
            except StopIteration:
                exhausted = True
            else:
                pending.add(asyncio.create_task(review_row(row)))
        if not pending:
            continue
        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            result, error = task.result()
            if error is not None:
                print(error, file=sys.stderr)
                failed += 1
            elif result is not None:
                append_jsonl(args.output, result)
                written += 1
    print(f"reviewed {written} item(s); {failed} failed")
    metadata_path = args.output.with_name(args.output.name + ".run.json")
    metadata: dict[str, object] = {"rubric_sha256": rubric_sha256(), "sessions": []}
    if metadata_path.exists():
        loaded = json.loads(metadata_path.read_text())
        if isinstance(loaded, dict) and isinstance(loaded.get("sessions"), list):
            metadata = loaded
    session_values = metadata["sessions"]
    if not isinstance(session_values, list):
        raise TypeError("review metadata sessions must be an array")
    sessions = cast(list[dict[str, object]], session_values)
    sessions.append(
        {
            "elapsed_seconds": round(time.monotonic() - session_started, 6),
            "items_written": written,
            "items_failed": failed,
            "items_skipped": len(completed),
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
        }
    )
    metadata["rubric_sha256"] = rubric_sha256()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return 1 if failed else 0


def _handle_review(args: argparse.Namespace) -> int:
    return asyncio.run(_review_rows(args))


def _handle_review_preparation(args: argparse.Namespace) -> int:
    candidate_paths: dict[str, Path] = {}
    for spec in args.candidate:
        condition, separator, raw_path = spec.partition("=")
        if not separator or not condition or not raw_path:
            raise ValueError("--candidate must use NAME=JSONL")
        if condition in candidate_paths:
            raise ValueError(f"duplicate candidate name {condition!r}")
        candidate_paths[condition] = Path(raw_path)
    if not 2 <= len(candidate_paths) <= 3:
        raise ValueError("exactly two or three --candidate values are required")
    resolved_outputs = {args.output.resolve(), args.mapping.resolve()}
    if len(resolved_outputs) != 2:
        raise ValueError("blinded output and private mapping paths must differ")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mapping.exists():
        raise FileExistsError(args.mapping)
    items, mappings = prepare_review_items(
        read_jsonl(args.sources),
        {condition: read_jsonl(path) for condition, path in candidate_paths.items()},
        target_language=args.language,
        task_type=args.task_type,
        id_field=args.id_field,
        source_field=args.source_field,
        output_field=args.output_field,
    )
    write_jsonl(args.output, items)
    write_jsonl(args.mapping, mappings)
    print(f"prepared {len(items)} blinded item(s); keep {args.mapping} private")
    return 0


def _handle_review_validation(args: argparse.Namespace) -> int:
    errors = validate_judge_batch(read_jsonl(args.results), read_jsonl(args.expected))
    if errors:
        for error in errors:
            print(error)
        return 1
    print("judge results are valid")
    return 0


def _handle_review_summary(args: argparse.Namespace) -> int:
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.native_review is not None and args.native_review.exists():
        raise FileExistsError(args.native_review)
    if args.native_review is not None and args.report.resolve() == args.native_review.resolve():
        raise ValueError("report and native-review paths must differ")
    report, native_items = build_review_report(
        read_jsonl(args.expected),
        read_jsonl(args.mapping),
        read_jsonl(args.results),
        force_languages=args.force_language,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.native_review is not None:
        write_jsonl(args.native_review, native_items)
    summary = cast(dict[str, object], report["summary"])["overall"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _handle_contract(args: argparse.Namespace) -> int:
    metadata_path = args.metadata or args.results.with_name(args.results.name + ".run.json")
    elapsed: float | None = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        sessions = metadata.get("sessions", []) if isinstance(metadata, dict) else []
        elapsed = sum(
            float(session["elapsed_seconds"])
            for session in sessions
            if isinstance(session, dict) and isinstance(session.get("elapsed_seconds"), int | float)
        )
    validation = validate_run(
        read_jsonl(args.sources),
        read_jsonl(args.results),
        id_field=args.id_field,
        elapsed_seconds=elapsed,
    )
    report = {"valid": validation.valid, **validation.summary, "issues": validation.issues}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation.summary, indent=2))
    if validation.issues:
        print(f"run contract failed with {len(validation.issues)} issue(s)")
        return 1
    print("run contract passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
