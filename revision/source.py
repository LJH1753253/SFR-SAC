"""Revision-extension source identity without changing the frozen main-study digest."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from v2.checkpoint import source_digest as main_source_digest


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def revision_source_digest() -> str:
    digest = hashlib.sha256()
    base = main_source_digest()
    digest.update(b"main-source:")
    digest.update(base.encode())
    for path in sorted((ROOT / "revision").glob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_revision_marker(directory: str | Path, expected: str | None = None) -> None:
    path = Path(directory) / "revision_source.json"
    path.write_text(
        '{\n'
        f'  "main_source_digest": "{main_source_digest()}",\n'
        f'  "revision_source_digest": "{expected or revision_source_digest()}"\n'
        '}\n',
        encoding="utf-8",
    )
