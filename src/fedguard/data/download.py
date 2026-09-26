"""Download the PhysioNet/CinC Challenge 2019 training data (open access, CC-BY 4.0) and verify it.

Primary source: the public S3 bucket ``physionet-open`` over plain HTTPS (anonymous ListObjectsV2 + GET),
so the AWS CLI is not required. Every object's ETag is its MD5 (single-part uploads), which lets us
verify each file, not only the counts. Fallback: the physionet.org directory index (size check only).

Expected counts: training_setA = 20,336 files, training_setB = 20,000 files. Anything else fails loudly.
"""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

S3_ENDPOINT = "https://physionet-open.s3.amazonaws.com"
S3_PREFIX = "challenge-2019/1.0.0/training"
PHYSIONET_BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
SUBSETS = ("training_setA", "training_setB")
EXPECTED_COUNTS = {"training_setA": 20336, "training_setB": 20000}
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class DownloadError(RuntimeError):
    """Raised when the data cannot be downloaded or fails verification."""


@dataclass(frozen=True)
class RemoteFile:
    subset: str
    name: str  # e.g. p000001.psv
    url: str
    size: int | None
    md5: str | None


def _get(session: requests.Session, url: str, retries: int = 5, **kw) -> requests.Response:
    """GET with exponential backoff; raises DownloadError after ``retries`` attempts."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            last = DownloadError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(min(30.0, 0.5 * 2**attempt))
    raise DownloadError(f"failed after {retries} attempts: {url} ({last})")


def list_s3(subset: str, session: requests.Session | None = None) -> list[RemoteFile]:
    """List all .psv objects of ``subset`` in the public bucket (paginated ListObjectsV2)."""
    s = session or requests.Session()
    out: list[RemoteFile] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": f"{S3_PREFIX}/{subset}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(_get(s, S3_ENDPOINT + "/", params=params).content)
        for c in root.findall("s3:Contents", _S3_NS):
            key = c.findtext("s3:Key", namespaces=_S3_NS) or ""
            if not key.endswith(".psv"):
                continue
            etag = (c.findtext("s3:ETag", namespaces=_S3_NS) or "").strip('"')
            out.append(
                RemoteFile(
                    subset=subset,
                    name=key.rsplit("/", 1)[-1],
                    url=f"{S3_ENDPOINT}/{key}",
                    size=int(c.findtext("s3:Size", namespaces=_S3_NS) or 0),
                    md5=etag if re.fullmatch(r"[0-9a-f]{32}", etag) else None,  # multipart ETags are not MD5s
                )
            )
        if (root.findtext("s3:IsTruncated", namespaces=_S3_NS) or "false") != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
    return out


def list_physionet(subset: str, session: requests.Session | None = None) -> list[RemoteFile]:
    """Fallback listing from the physionet.org HTML index (no checksums available)."""
    s = session or requests.Session()
    html = _get(s, f"{PHYSIONET_BASE}/{subset}/").text
    names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
    return [RemoteFile(subset, n, f"{PHYSIONET_BASE}/{subset}/{n}", None, None) for n in names]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _is_valid(path: Path, rf: RemoteFile, check_md5: bool) -> bool:
    if not path.exists():
        return False
    if rf.size is not None and path.stat().st_size != rf.size:
        return False
    return not (check_md5 and rf.md5 is not None and _md5(path) != rf.md5)


def _fetch(rf: RemoteFile, dest: Path, session: requests.Session, check_md5: bool) -> str:
    if _is_valid(dest, rf, check_md5):
        return "skipped"
    content = _get(session, rf.url).content
    if rf.size is not None and len(content) != rf.size:
        raise DownloadError(f"size mismatch for {rf.name}: got {len(content)}, expected {rf.size}")
    if rf.md5 is not None and hashlib.md5(content, usedforsecurity=False).hexdigest() != rf.md5:
        raise DownloadError(f"MD5 mismatch for {rf.name}")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(dest)
    return "downloaded"


def select_subset(names: Iterable[str], n: int | None, seed: int) -> list[str]:
    """Deterministic seeded subset of file names (sorted first, so it is independent of listing order)."""
    from fedguard.utils.seed import rng_for

    names = sorted(names)
    if n is None or n >= len(names):
        return names
    idx = rng_for(seed, "fast-subset").choice(len(names), size=n, replace=False)
    return sorted(names[i] for i in idx)


def download(
    raw_dir: Path,
    subset_per_hospital: int | None = None,
    seed: int = 42,
    workers: int = 32,
    check_md5: bool = True,
    source: str = "s3",
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict[str, int]]:
    """Download (or resume) the training data into ``raw_dir/<subset>/``.

    With ``subset_per_hospital`` only a deterministic seeded subset is fetched (``--fast``).
    Returns per-subset counts of downloaded / skipped files.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    stats: dict[str, dict[str, int]] = {}
    for subset in SUBSETS:
        try:
            files = list_s3(subset, session) if source == "s3" else list_physionet(subset, session)
        except DownloadError:
            if source != "s3":
                raise
            files = list_physionet(subset, session)
        wanted = set(select_subset((f.name for f in files), subset_per_hospital, seed))
        files = [f for f in files if f.name in wanted]
        out_dir = Path(raw_dir) / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        counts = {"downloaded": 0, "skipped": 0}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_fetch, f, out_dir / f.name, session, check_md5) for f in files]
            for i, fut in enumerate(as_completed(futs), 1):
                counts[fut.result()] += 1
                if progress and (i % 500 == 0 or i == len(futs)):
                    progress(subset, i, len(futs))
        stats[subset] = counts
    return stats


def verify(
    raw_dir: Path, expected: dict[str, int] | None = EXPECTED_COUNTS, min_size: int = 100
) -> dict[str, int]:
    """Check file counts per subset (exact match to ``expected`` unless it is None) and that no file is
    truncated or unreadable. Raises DownloadError with an actionable message on failure."""
    counts: dict[str, int] = {}
    problems: list[str] = []
    for subset in SUBSETS:
        d = Path(raw_dir) / subset
        files = sorted(d.glob("*.psv")) if d.exists() else []
        counts[subset] = len(files)
        if expected is not None and len(files) != expected[subset]:
            problems.append(f"{subset}: found {len(files)} .psv files, expected {expected[subset]}")
        tiny = [f.name for f in files if f.stat().st_size < min_size]
        if tiny:
            problems.append(f"{subset}: {len(tiny)} suspiciously small files, e.g. {tiny[:3]}")
        partial = list(d.glob("*.part")) if d.exists() else []
        if partial:
            problems.append(f"{subset}: {len(partial)} incomplete .part files (re-run the download)")
    if problems:
        raise DownloadError(
            "PhysioNet 2019 data failed verification:\n  - "
            + "\n  - ".join(problems)
            + f"\nRe-run `fedguard data download` (it resumes). Raw dir: {raw_dir}"
        )
    return counts
