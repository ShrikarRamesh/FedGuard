"""Back up experiment runs (and, once, the raw data) as single zip archives into a synced folder.

    python scripts/backup.py [--dest <dir>] [--keep 3] [--no-raw]

* runs/ -> <dest>/runs_<timestamp>.zip (checkpoints, predictions, events, logs, GPU log). The newest --keep archives are
  kept. Runs still in progress are captured as they are at that moment (the next snapshot supersedes them).
* data/raw/physionet2019 -> <dest>/data_raw_physionet2019.zip, only if it does not exist yet (the raw files never change;
  they can also be re-downloaded with `fedguard data download`). data/processed is NOT backed up: it is regenerated
  deterministically by `fedguard data process` (~20 s).
* <dest>/MANIFEST.json records size, file count and SHA-256 of every archive.
Default dest: <OneDrive>/FedGuard_backups (single archives avoid the per-file sync churn of mirroring runs/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

from fedguard.utils.io import data_dir, runs_dir


def zip_dir(src: Path, dest_zip: Path) -> dict:
    tmp = dest_zip.with_suffix(".zip.part")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and not p.name.endswith((".part", ".tmp")) and p.name != "_slots.lock":
                try:
                    z.write(p, p.relative_to(src.parent))
                    n += 1
                except (PermissionError, OSError):
                    continue  # file being written right now; captured by the next snapshot
    tmp.replace(dest_zip)
    h = hashlib.sha256()
    with dest_zip.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"file": dest_zip.name, "files": n, "bytes": dest_zip.stat().st_size, "sha256": h.hexdigest(),
            "created": datetime.now().isoformat(timespec="seconds"), "source": str(src)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path(os.environ.get("OneDrive", Path.home())) / "FedGuard_backups")
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    man_path = a.dest / "MANIFEST.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"archives": []}
    entry = zip_dir(runs_dir(), a.dest / f"runs_{datetime.now():%Y%m%d-%H%M%S}.zip")
    manifest["archives"].append(entry)
    print(f"runs -> {entry['file']}: {entry['files']} files, {entry['bytes'] / 2**20:.0f} MiB, sha256 {entry['sha256'][:16]}...")
    raw = data_dir() / "raw" / "physionet2019"
    raw_zip = a.dest / "data_raw_physionet2019.zip"
    if not a.no_raw and raw.exists() and not raw_zip.exists():
        e = zip_dir(raw, raw_zip)
        manifest["archives"].append(e)
        print(f"raw data -> {e['file']}: {e['files']} files, {e['bytes'] / 2**20:.0f} MiB")
    runs_zips = sorted(a.dest.glob("runs_*.zip"))
    for old in runs_zips[: max(0, len(runs_zips) - a.keep)]:
        old.unlink()
        print(f"removed old snapshot {old.name} (keeping {a.keep})")
    manifest["archives"] = [m for m in manifest["archives"] if (a.dest / m["file"]).exists()]
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"backup dir: {a.dest}")


if __name__ == "__main__":
    main()
