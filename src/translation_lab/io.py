import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def write_jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def jsonl_ids(path: Path, id_field: str) -> set[object]:
    return {row[id_field] for row in read_jsonl(path) if id_field in row}
