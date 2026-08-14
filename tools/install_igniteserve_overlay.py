#!/usr/bin/env python3
"""Install or restore the IgniteServe Python-only PyTorch 2.7 overlay."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


RUNTIME_FILES = (
    "_inductor/compile_fx_full_subproc.py",
    "_inductor/compile_fx_ext.py",
    "_inductor/compile_fx_subproc.py",
    "fx/_graph_pickler.py",
)
BACKUP_ROOT_NAME = ".igniteserve_overlay_backups"
MANIFEST_NAME = "install_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="Target Python interpreter")
    parser.add_argument("--restore", action="store_true",
                        help="Restore the latest overlay backup")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def inspect_target(python: str) -> tuple[str, Path]:
    code = (
        "import json, pathlib, torch; "
        "print(json.dumps({'version': torch.__version__, "
        "'root': str(pathlib.Path(torch.__file__).resolve().parent)}))"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [python, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        cwd=Path.home(),
        env=environment,
    )
    data = json.loads(result.stdout.strip().splitlines()[-1])
    return data["version"], Path(data["root"])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def repository_torch_root() -> Path:
    root = Path(__file__).resolve().parents[1] / "torch"
    if not all((root / name).is_file() for name in RUNTIME_FILES):
        raise RuntimeError(f"Incomplete IgniteServe source tree: {root}")
    return root


def latest_backup(torch_root: Path) -> Path:
    root = torch_root / BACKUP_ROOT_NAME
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()), reverse=True
    ) if root.is_dir() else []
    if not candidates:
        raise RuntimeError(f"No IgniteServe overlay backup found under {root}")
    return candidates[0]


def install(args: argparse.Namespace, version: str, torch_root: Path) -> None:
    if not version.startswith("2.7.") and not args.allow_version_mismatch:
        raise RuntimeError(
            f"Expected PyTorch 2.7.*, found {version}. "
            "Use --allow-version-mismatch only after source-level validation."
        )

    source_root = repository_torch_root()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = torch_root / BACKUP_ROOT_NAME / stamp
    manifest: dict[str, object] = {
        "schema_version": 1,
        "installed_at_utc": stamp,
        "torch_version": version,
        "torch_root": str(torch_root),
        "files": [],
    }

    for relative in RUNTIME_FILES:
        source = source_root / relative
        target = torch_root / relative
        if relative != "_inductor/compile_fx_full_subproc.py" and not target.is_file():
            raise RuntimeError(f"Expected PyTorch module is missing: {target}")
        entry = {
            "relative_path": relative,
            "source_sha256": digest(source),
            "original_exists": target.exists(),
            "original_sha256": digest(target) if target.exists() else None,
        }
        manifest["files"].append(entry)

    print(f"PyTorch {version}: {torch_root}")
    print(f"Overlay backup: {backup}")
    if args.dry_run:
        for entry in manifest["files"]:
            print(f"would install: {entry['relative_path']}")
        return

    backup.mkdir(parents=True, exist_ok=False)
    for entry in manifest["files"]:
        relative = str(entry["relative_path"])
        source = source_root / relative
        target = torch_root / relative
        if target.exists():
            backup_target = backup / relative
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (backup / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("IgniteServe PyTorch overlay installed.")


def restore(args: argparse.Namespace, version: str, torch_root: Path) -> None:
    backup = latest_backup(torch_root)
    manifest_path = backup / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["torch_root"]).resolve() != torch_root.resolve():
        raise RuntimeError("Backup belongs to a different torch installation")
    print(f"PyTorch {version}: {torch_root}")
    print(f"Restoring backup: {backup}")
    if args.dry_run:
        return

    for entry in manifest["files"]:
        relative = str(entry["relative_path"])
        target = torch_root / relative
        saved = backup / relative
        if entry["original_exists"]:
            if not saved.is_file():
                raise RuntimeError(f"Backup file is missing: {saved}")
            shutil.copy2(saved, target)
        elif target.exists():
            target.unlink()
    print("Original PyTorch modules restored.")


def main() -> int:
    args = parse_args()
    version, torch_root = inspect_target(args.python)
    if args.restore:
        restore(args, version, torch_root)
    else:
        install(args, version, torch_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
