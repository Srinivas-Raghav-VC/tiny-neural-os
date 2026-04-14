# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///
"""Validate marimo session JSON files for notebooks/.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from marimo._server.file_router import AppFileRouter
from marimo._server.files.directory_scanner import is_marimo_app
from marimo._utils.files import expand_file_patterns
from marimo._utils.marimo_path import MarimoPath

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
ERROR_PATTERNS = ["ModuleNotFoundError"]


def report_error(file: str, msg: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error file={file}::{msg}")
    else:
        print(f"  ERROR: {msg}")


def get_session_path(notebook_path: Path) -> Path:
    return notebook_path.parent / "__marimo__" / "session" / f"{notebook_path.name}.json"


def find_notebooks() -> list[Path]:
    all_files = expand_file_patterns((str(NOTEBOOKS_DIR),))
    return [Path(f) for f in all_files if is_marimo_app(str(f))]


def hash_code(code: str) -> str:
    return hashlib.md5(code.encode("utf-8"), usedforsecurity=False).hexdigest()


def main() -> None:
    notebooks = find_notebooks()
    if not notebooks:
        print("No marimo notebooks found.")
        return

    ok = True
    for nb in notebooks:
        rel_nb = nb.relative_to(REPO_ROOT)
        session = get_session_path(nb)
        if not session.exists():
            report_error(str(rel_nb), f"Missing session JSON: {session.relative_to(REPO_ROOT)}")
            ok = False
            continue

        content = session.read_text()
        for pattern in ERROR_PATTERNS:
            if pattern in content:
                report_error(str(rel_nb), f"Found '{pattern}' in {session.relative_to(REPO_ROOT)}")
                ok = False

        marimo_path = MarimoPath(str(nb))
        file_router = AppFileRouter.from_filename(marimo_path)
        file_key = file_router.get_unique_file_key()
        assert file_key is not None
        file_manager = file_router.get_file_manager(file_key)
        cell_manager = file_manager.app.cell_manager
        notebook_hashes = Counter(
            hash_code(cell_manager.get_cell_data(cid).code)
            for cid in cell_manager.cell_ids()
        )
        session_hashes = Counter(c["code_hash"] for c in json.loads(content).get("cells", []))
        if notebook_hashes != session_hashes:
            report_error(str(rel_nb), f"Session JSON is out of sync for {rel_nb}")
            ok = False

    if not ok:
        sys.exit(1)
    print("All session files are present and up to date.")


if __name__ == "__main__":
    main()
